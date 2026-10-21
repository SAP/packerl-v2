import subprocess
import os
import time
from typing import List, Dict, Any

from omegaconf import DictConfig
import torch
from torch import Tensor
import numpy as np

import py_interface as ns3ai
import networkx as nx
from torch_geometric.utils import degree
from torch_scatter import scatter_max, scatter_add

from .scenario import NetworkScenario
from .scenario.events import Event as PSEvent
from .utils.actions import PackerlEnvAction
from .utils.canonical_bytes import canonical_bytes, canonical_hash
from .utils.constants import (NS3_DIR,
                              SHM_SIZE,
                              ETHERNET_MTU,
                              IPV4_HEADER_SIZE,
                              UDP_HEADER_SIZE,
                              TCP_HEADER_SIZE,
                              BIRDSEYE_OBSERVER_ID)
from .utils.features import (ALL_GLOBAL_FEATURES,
                             global_feat_in_struct_order,
                             node_feat_in_struct_order,
                             edge_feat_in_struct_order,
                             edge_delay_idx)
from .utils.logging import (LOG_CONSOLE_FUNCTION,
                            LOG_CONSOLE_INFO,
                            LOG_CONSOLE_RUN,
                            LOG_CONSOLE_DEBUG,
                            LOG_CONSOLE_LEVEL_AT_MOST,
                            LOG_CONSOLE_UNCOND,
                            finalize_console_logger,
                            finalize_experiment_recorder)
from .utils.observations import ObservationGraph, Observation
from .utils.time import TimedBlock
from .utils.types import TrafficMatrix, StepResult, ActionSedi, Reward
from .utils.shared_memory.interaction import place_shm, read_shm, check_shm
from .utils.shared_memory.structs import PackerlEnvStruct, PackerlActStruct, ALL_SHM_SIZES
from .utils.sp_calculator import DelayShortestPathCalculator


class PackerlEnv:
    """
    This environment is an interface for learning routing optimization in computer networks,
    using the packet-level network simulator ns-3 (https://www.nsnam.org) as a backend.
    The environment provides network state views in the form of their current topology as well
    as network configuration (e.g. delays, datarates) and current load and performance
    metrics such as link utilization and queue load, or packet loss/avg. packet delay.
    The agents/policy shall provide routing actions for each participating router/node that
    get installed in the simulation and used as consultation for routing table entries
    for the upcoming timestep. The reward correlates with the overall network performance,
    including metrics such as minimizing dropped packets, maximizing network throughput
    or data received rate, or minimizing average packet delay.

    This environment uses the 'scenario' package to provide a wide range of different
    network scenarios to represent the diversity of networking conditions encountered
    in the real world.
    On each reset() call, it uses a scenario generator to provide a new network scenario
    (topology, configuration and traffic) which is installed on the interfaced ns-3 simulation.

    This environment is designed to be used as a scoped context manager, because certain
    resources (e.g. shared memory) need to be freed after the simulation is done.
    """
    def __init__(self, config: DictConfig):
        """
        Initializes the environment, which includes setting up the shared memory interface,
        the ns-3 simulation, the data generator, the reward module and some bookkeeping.
        """
        self.config = config

        LOG_CONSOLE_RUN("=== initializing PackerlEnv ===")

        self.comm_delay_provider = DelayShortestPathCalculator()

        # ns-3 interfacing; create shared memory pool for communication with ns-3
        LOG_CONSOLE_DEBUG(f"shm sizes: {ALL_SHM_SIZES}")
        if self.config.mempool_key < 1001:
            raise ValueError("mempool_key has to be larger than 1000")
        ns3ai.Init(self.config.mempool_key, SHM_SIZE)
        LOG_CONSOLE_INFO("initialized shared memory pool")
        self.experiment = ns3ai.Experiment(self.config.mempool_key,
                                           SHM_SIZE,
                                           'packerl',
                                           NS3_DIR,
                                           build_ns3=False)
        self.sim = None  # initialized in reset()
        self.ns3_proc = None  # process object for the ns-3 simulation, initialized in reset() and used for debugging
        LOG_CONSOLE_INFO("created ns3ai experiment")

        self.cur_scenario: NetworkScenario | None = None  # current scenario, provided in reset()
        self.in_episode = False
        self.episode_has_just_begun = False  # set to True right after env reset; set to False after first step
        self.t = 0
        self.chance_to_attach_gdb_given = False

    def __enter__(self):
        """
        Nothing special to do here, just return self
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Properly deletes the member attributes 'sim' and 'experiment' and calls
        a bash script to free shared memory.
        """
        LOG_CONSOLE_INFO("deleting PackerlEnv...")
        del self.sim
        del self.experiment
        subprocess.run(os.path.join(NS3_DIR, "contrib/ns3-ai/freeshm.sh"), check=True)

        LOG_CONSOLE_INFO("finalizing loggers...")
        finalize_experiment_recorder()
        finalize_console_logger()

    @property
    def observation(self) -> Observation:
        """
        :return: the latest observation (filtered)
        """
        if len(self._observations) < 1:
            raise RuntimeError("observation() got called before any observation was received, "
                               "which should not happen!")
        return self._observations[-1]

    @property
    def action(self) -> PackerlEnvAction:
        """
        Returns: the latest action (all agents)
        """
        if len(self._actions) < 1:
            raise RuntimeError("action() got called before any action was received, "
                               "which should not happen!")
        return self._actions[-1]

    @property
    def reward(self) -> Reward:
        """
        :return: the latest reward.
        """
        if len(self._rewards) < 1:
            raise RuntimeError("reward() got called before any reward was received, "
                               "which should not happen!")
        return self._rewards[-1]

    @property
    def upcoming_events(self) -> List[PSEvent]:
        """
        :return: the events that are scheduled for the upcoming time step
        """
        events = self.cur_scenario.events
        events_horizon = len(events)
        if self.t > events_horizon:
            raise ValueError(f"current time step t ({self.t}) exceeds scenario events length ({events_horizon})")
        elif self.t == events_horizon:
            return []
        else:
            return events[self.t]

    @property
    def past_traffic_matrix(self) -> TrafficMatrix:
        """
        :return: the traffic matrix observed in the past time step
        """
        if len(self._traffic_matrices) < 1:
            raise RuntimeError("past_traffic_matrix() got called before any traffic was observed, "
                               "which should not happen!")
        return self._traffic_matrices[-1]

    @property
    def comm_delays_from_central_node(self) -> (Tensor, Tensor):
        """
        :return: the communication delay entries from the central node to all other nodes,
                 as (dests, delays) 1D tensors.
        """
        comm_delays_from_central_node: np.ndarray = self.comm_delays[self.comm_delays[:, 0] == self.central_node]
        dests = torch.from_numpy(comm_delays_from_central_node[:, 1]).long()
        delays = torch.from_numpy(comm_delays_from_central_node[:, 2]).float()
        return dests, delays

    @staticmethod
    def _create_mock_obs_graph(network: nx.Graph) -> ObservationGraph:
        """
        Creates a mock ObservationGraph from the given networkx graph, only containing the topology
        and edge delays.
        :param network: The networkx graph representing the network topology.
        :return: The mock ObservationGraph.
        """
        sorted_nodes = sorted(network.nodes)
        directed_edge_list = [
            (n1, n2, data.copy())
            for u, v, data in network.edges(data=True)
            for (n1, n2) in ((u, v), (v, u))
        ]
        directed_edge_list.sort(key=lambda x: (x[0], x[1]))
        topology_directed = nx.DiGraph()
        topology_directed.add_nodes_from(sorted_nodes)
        topology_directed.add_edges_from(directed_edge_list)

        # assemble information for converting topology graph to mock PyG data ObservationGraph
        edge_delays = torch.tensor([int(topology_directed[u][v]["delay"]) for u, v in topology_directed.edges])  # shape: (num_directed_edges,)
        mock_edge_attr = torch.empty(topology_directed.number_of_edges(), len(edge_feat_in_struct_order))  # excl. mock linkWeight, since that is not present in obs
        mock_edge_attr[:, edge_delay_idx] = edge_delays  # put actual delays in the delay feature
        mock_agent_index = torch.tensor(list(topology_directed.edges), dtype=torch.long).t().contiguous()  # shape: (2, num_edges)
        mock_agent_src, mock_agent_dst = mock_agent_index
        mock_agent_ids = torch.tensor(list(sorted_nodes), dtype=torch.long)
        mock_edge_src = torch.searchsorted(mock_agent_ids, mock_agent_src)
        mock_edge_dst = torch.searchsorted(mock_agent_ids, mock_agent_dst)
        mock_edge_index = torch.stack([mock_edge_src, mock_edge_dst], dim=0)  # shape: (2, num_edges)
        mock_obs = ObservationGraph(x=torch.empty(len(sorted_nodes), len(node_feat_in_struct_order)),
                                    edge_index=mock_edge_index,
                                    edge_attr=mock_edge_attr,
                                    u=torch.empty(1, len(global_feat_in_struct_order)),
                                    agent_ids=mock_agent_ids,
                                    observer_ids=torch.tensor([BIRDSEYE_OBSERVER_ID], dtype=torch.long),
                                    )
        return mock_obs

    def reset(self, scenario: NetworkScenario) -> Observation:
        """
        Resets the simulator, creates a new scenario and does the necessary bookkeeping.
        :return: Network observation prior to the first action.
        """

        # bookkeeping
        self.t = 0
        self._observations = []
        self._traffic_matrices = []
        self._actions = []
        self._node_embeddings = []
        self._rewards = []

        # use suitable data generator to create scenario for this episode
        self.cur_scenario = scenario
        if LOG_CONSOLE_LEVEL_AT_MOST("info"):
            scenario_hash, topology_hash, events_hash = self.cur_scenario.get_hash_strs()
            LOG_CONSOLE_INFO(f"starting episode, scenario hash: {scenario_hash} "
                             f"(topology: {topology_hash}, events: {events_hash})")

        # obtain communication delays and central node from the initial topology graph
        mock_obs_graph = self._create_mock_obs_graph(self.cur_scenario.network)
        self.comm_delays, self.central_node = self.comm_delay_provider.get_node_distances_and_central_node(mock_obs_graph)

        # initialize and start ns-3 simulator, and link shared memory block with ns3
        self.experiment.reset()
        # if implementing vectorized envs in the future, we can use range(0, num_envs-1) for 'memblock_key' and 'run'
        ns3_run_id = 0  # acts as rng seed for ns3 and thus ensures that same scenarios lead to same results
        memblock_key = 0  # distinguishes shared memory blocks of different envs in case of vectorized envs
        local_reward_config = self.config.trainer.ppo.reward.local_rewards
        self.sim = ns3ai.Ns3AIRL(uid=memblock_key, EnvType=PackerlEnvStruct, ActType=PackerlActStruct)
        experiment_setting = {
            "run": ns3_run_id,
            "memblockKey": memblock_key,
            "mtu": ETHERNET_MTU,
            "ipv4HeaderSize": IPV4_HEADER_SIZE,
            "udpHeaderSize": UDP_HEADER_SIZE,
            "tcpHeaderSize": TCP_HEADER_SIZE,
            "birdseyeObserverId": BIRDSEYE_OBSERVER_ID,
            "localRewardSpatialDecay": local_reward_config.spatial_decay,
            "localRewardPathCutoff": local_reward_config.path_cutoff,
            "localRewardCapToBytes": local_reward_config.cap_to_bytes,
            "simStepDuration": self.cur_scenario.ms_per_step,
            "bufferSizeBdp": self.cur_scenario.buffer_size_bdp,
            "probTcp": self.cur_scenario.prob_tcp,
            "useTcpSack": self.cur_scenario.use_tcp_sack,
            "capLinkUtilization": False,  # link utilization values can exceed 1.0 due to the way we calculate the maximum transferrable bytes per step - we don't cap them in ns3-packerl
            "probabilisticRouting": False,  # currently not implemented
        }
        LOG_CONSOLE_INFO(f"mempool key: {hex(self.config.mempool_key)} {memblock_key=}")
        self.ns3_proc = self.experiment.run(setting=experiment_setting,
                                            show_output=True,
                                            log_modules=self.config.logging.ns3_log_modules,
                                            log_level=self.config.logging.log_level,
                                            profile_ns3=self.config.profiling_cpp
                                            )

        # check whether ns3 is ready to interact via shared memory
        check_shm(self.sim)
        LOG_CONSOLE_INFO(f"started sim experiment")

        # give user a chance to attach debugger to ns3 process
        if self.config.attach_gdb and not self.chance_to_attach_gdb_given:
            time.sleep(0.5)  # this just makes the following line appear at the bottom of your console
            LOG_CONSOLE_UNCOND(f"\nGiving you a chance to attach your debugger to the ns3 process (10 seconds).")
            time.sleep(10)
            self.chance_to_attach_gdb_given = True  # only give the user a chance to attach once

        # Initialize episode interaction by communicating network graph and comm. delays, and getting first observation
        place_shm(self.sim, "network_graph", self.cur_scenario.network, self.ns3_proc)
        place_shm(self.sim, "global_reward_weights", self.config.trainer.ppo.reward.global_rewards, self.ns3_proc)
        place_shm(self.sim, "communication_delays", (self.comm_delays, self.central_node), self.ns3_proc)
        if self.sim.isFinish():
            raise RuntimeError("ns-3 simulation finished before it even started. This should not happen.")
        self._observe(store_rewards=False)

        # finalize
        self.in_episode = True
        self.episode_has_just_begun = True
        return self.observation

    def step(self, action_sedi: ActionSedi, action_delays_per_node: Tensor, node_emb: Tensor | None) -> StepResult:
        """
        This env's step method:
         1. places the given actions, the latest events, and the latest communication delays into the shared memory
         2. Waits for ns-3 to install this info and simulate the network for the specified amount of time,
         3. Receives the network observations from ns-3 via shared memory and processes it.

        :param action_sedi: The actions to be taken, provided as selected edge destination indices (sedi).
            These will be converted into graph shape using the past observation as base topology.
        :param action_raw: The raw actions provided by the algorithm. They may be link weights, in which case we store
            them for the next observation.
        :param action_delays_per_node: Action delays per agent: the installation of the provided actions in ns-3
            will be delayed by these amounts (in ms) per node.

        :return: A step result tuple containing the observation (incl. reward), done flag, and additional step info.
        """
        LOG_CONSOLE_FUNCTION()
        if not self.in_episode:
            raise RuntimeError("Can't run step() if experiment is not running. Call reset() first.")

        # convert provided action sedi into graph shape, using the past observation as base topology.
        # Since the sedi_block still is in inner node ID representation,
        # we also supply the agent_ids to convert to routing actions into agent ID representation
        past_global_obs_graph = self.observation.get_birdseye_obs_graph()
        ac_edge_index = past_global_obs_graph.edge_index.clone()
        agent_ids = past_global_obs_graph.agent_ids.clone()
        N, E = past_global_obs_graph.num_nodes, past_global_obs_graph.num_edges
        sedi_block = torch.zeros(E * N, dtype=torch.float32).scatter_(0, action_sedi, 1).reshape(E, N)
        actions = PackerlEnvAction(edge_index=ac_edge_index, edge_attr=sedi_block, agent_ids=agent_ids)

        with TimedBlock() as step_timer:

            # place upcoming events (traffic flow arrivals, link failures etc.) and actions in shared memory.
            # After ns-3 has consumed the placed info completely, it runs a simulation timestep.
            place_shm(self.sim, "events", self.upcoming_events, self.ns3_proc)
            place_shm(self.sim, "actions", (actions, action_delays_per_node), self.ns3_proc)
            place_shm(self.sim, "communication_delays", (self.comm_delays, self.central_node), self.ns3_proc)

            # ========================================
            # ... meanwhile, ns3 runs the sim step ...
            # ========================================

            self._actions.append(actions.detach().cpu())
            if node_emb is not None:
                node_emb_agent_dict = {agent_ids[i].item(): (node_emb[i, 0].item(), node_emb[i, 1].item())
                                       for i in range(node_emb.shape[0])}
                self._node_embeddings.append(node_emb_agent_dict)
            self.t += 1

            # obtain results. This will block until ns3 has finished the sim step and has written
            # the results to the shared memory block. Then, we obtain the observation, reward (global and local)
            # and traffic matrices for the current timestep, and store them in the respective lists.
            self._observe()

        # actual step is finished -> process results. there is no terminal state,
        #  our ns3 sim is cut off after the last step (i.e. truncated, never terminated)
        obs = self.observation
        done = self.t >= self.cur_scenario.length
        reward = self.reward
        r_values, r_observer_ids = reward
        global_reward = r_values[r_observer_ids == BIRDSEYE_OBSERVER_ID]
        local_reward = r_values[r_observer_ids != BIRDSEYE_OBSERVER_ID]
        infos = {
            "done": done,
            "env_step_time_ms": step_timer.elapsed_ms,
            "global_reward": global_reward.item(),
            "local_reward": local_reward.tolist(),
        }
        place_shm(self.sim, "done", done, self.ns3_proc)
        LOG_CONSOLE_INFO(f"PackerlEnv.step(): t={self.t}, global_r={global_reward.item()}")

        # handle end of episode
        if done:
            self.in_episode = False

            # obtain episode statistics
            oscillation_ratio, next_hop_spread = self._get_ep_action_stats()

            # create episode summary
            ep_summary = {
                "ep_scenario_stats": self.cur_scenario.get_stats(),
                "ep_metrics": self._get_ep_metrics(),
                "oscillation_ratio_global": oscillation_ratio.mean().item(),
                "next_hop_spread_global": next_hop_spread.mean().item(),
            }
            if self.cur_scenario.visualize:
                ep_summary["ep_vis_material"] = {
                    "scenario_topology": self.cur_scenario.network,
                    "scenario_events": self.cur_scenario.events,
                    "graph_name": self.cur_scenario.network.graph['name'],
                    "graph_node_pos": nx.get_node_attributes(self.cur_scenario.network, "pos"),
                    "obs_graphs": [obs.get_birdseye_obs_graph() for obs in self._observations],
                    "actions": self._actions,
                    "oscillation_ratio": oscillation_ratio.numpy(),
                    "next_hop_spread": next_hop_spread.numpy(),
                    "traffic_matrices": self._traffic_matrices,
                    "ep_step_stats": self._get_ep_step_stats(),
                }
                if len(self._node_embeddings) > 0:
                    ep_summary["ep_vis_material"]["node_emb"] = self._node_embeddings
            infos.update(ep_summary=ep_summary)
            if LOG_CONSOLE_LEVEL_AT_MOST("info"):
                LOG_CONSOLE_INFO(f"completed episode, trajectory hash: {self._episode_to_hash_str()}")

        self.episode_has_just_begun = False
        return obs, reward, done, infos

    def _observe(self, store_rewards: bool = True):
        """
        This method is called after each step to obtain the latest observation and reward from ns-3 via shared memory.
        """
        LOG_CONSOLE_FUNCTION()

        # obtain results from shared memory
        # TODO: we're providing the scenario's initial topology's V and E as upper bounds for the buffer sizes.
        #  This will break once we're adding nodes/edges during simulation -> provide actual current V, E instead.
        shm_content = read_shm(self.cur_scenario.network.number_of_nodes(),
                               self.cur_scenario.network.number_of_edges() * 2,
                               self.sim,
                               self.comm_delay_provider,
                               self.ns3_proc)
        (obs,
         global_traffic_matrix,
         _,  # central_node_traffic_matrix unused
         reward,
         _,  # monitoring_time unused
         _,  # elapsed_time unused
         comm_delays,
         central_node) = shm_content

        # store one-hot encoding of scenario and timestep in observation
        s_onehot = torch.from_numpy(self.cur_scenario.onehot_id)
        t_onehot = torch.zeros(self.cur_scenario.length + 1, dtype=torch.float32)
        t_onehot[self.t] = 1.0
        obs.s_t_onehot = torch.cat([s_onehot, t_onehot], dim=0).unsqueeze(0)  # shape: (1, S + T)

        # store observations
        self._observations.append(obs)
        self.comm_delays = comm_delays
        self.central_node = central_node
        self._traffic_matrices.append(global_traffic_matrix)  # TODO central_node_traffic_matrix unused
        if store_rewards:
            self._rewards.append(reward)

    def _get_ep_metrics(self) -> Dict[str, Any]:
        """
        :return: A dict containing the episode's aggregated performance metrics
        """
        ep_metrics = {}
        auxiliary_metrics = []

        # iterate over all global observations and aggregate the metrics. Some need special treatment (see below).
        for obs in self._observations:
            global_values = obs.get_birdseye_obs_graph().u.flatten().tolist()

            if len(global_values) != len(global_feat_in_struct_order):
                raise ValueError("global observation has wrong number of features (expected: "
                                 f"{len(global_feat_in_struct_order)}, got: {len(global_values)})")

            for metric_name, value in zip(global_feat_in_struct_order, global_values):
                metric_type = ALL_GLOBAL_FEATURES[metric_name][1]

                # cumulative sum
                if metric_type in "bca":  # byte/channel/auxiliary metrics
                    if metric_name == "inFlight": continue  # inFlight is a momentary snapshot derived from cumulative metrics
                    if metric_name not in ep_metrics:
                        ep_metrics[metric_name] = value
                    else:
                        ep_metrics[metric_name] += value
                    if metric_type == "a":
                        auxiliary_metrics.append(metric_name)

                # max
                elif metric_name.startswith("max"):  # maxDelay, maxLinkUtilization, maxTxQueueRelativeLoad
                    if metric_name not in ep_metrics or value > ep_metrics[metric_name]:
                        ep_metrics[metric_name] = value

                # weighted cumulative sum
                elif metric_type == "d" and metric_name.startswith("avg"):  # avgDelay, avgTcpSenderSrtt, avgJitter
                    count_name = metric_name[3].lower() + metric_name[4:] + "Count"  # e.g. avgTcpSenderSrtt -> tcpSenderSrttCount
                    weighted_value = global_values[global_feat_in_struct_order.index(count_name)] * value
                    if metric_name not in ep_metrics:
                        ep_metrics[metric_name] = weighted_value
                    else:
                        ep_metrics[metric_name] += weighted_value

        # 'avg' d-type metrics: divide by ep-wide count sum to get ep-wide average
        for metric_name, value in ep_metrics.items():
            if metric_name.startswith("avg"):
                count_name = metric_name[3].lower() + metric_name[4:] + "Count"
                count_value = ep_metrics[count_name]
                if count_value > 0:
                    ep_metrics[metric_name] /= count_value
                else:
                    ep_metrics[metric_name] = 0.0

        # calculate remaining ep-wide metrics
        given_up_or_delivered = ep_metrics['givenUp'] + ep_metrics['delivered']
        ep_metrics["inFlight"] = ep_metrics["tx"] - given_up_or_delivered
        ep_metrics["dropRatio"] = ep_metrics["givenUp"] / given_up_or_delivered if given_up_or_delivered > 0 else 0.0
        ep_metrics["inFlightRatio"] = ep_metrics["inFlight"] / ep_metrics["tx"] if ep_metrics["tx"] > 0 else 0.0
        ep_metrics["avgLinkUtilization"] \
            = ep_metrics["transmitted"] / ep_metrics["transmittable"] if ep_metrics["transmittable"] > 0 else 0.0
        ep_metrics["avgTxQueueRelativeLoad"] = ep_metrics["txQueueLoad"] / ep_metrics["txQueueCapacity"] \
            if ep_metrics["txQueueCapacity"] > 0 else 0.0

        for aux_metric in auxiliary_metrics:
            ep_metrics.pop(aux_metric, None)

        return ep_metrics

    def _get_ep_step_stats(self) -> Dict[str, Any]:
        """
        :return: A dict containing the episode's step-wise statistics
        """
        LOG_CONSOLE_FUNCTION()

        cumsum_metrics = []
        ep_step_stats = {}

        for obs in self._observations:
            global_values = obs.get_birdseye_obs_graph().u.flatten().tolist()

            if len(global_values) != len(global_feat_in_struct_order):
                raise ValueError("global observation has wrong number of features (expected: "
                                 f"{len(global_feat_in_struct_order)}, got: {len(global_values)})")

            for metric_name, value in zip(global_feat_in_struct_order, global_values):
                metric_type = ALL_GLOBAL_FEATURES[metric_name][1]

                if metric_type in "bcdurl":
                    if metric_name not in ep_step_stats:
                        ep_step_stats[metric_name] = []

                    ep_step_stats[metric_name].append(value)

                    if metric_type in "bc":
                        cumsum_metrics.append(metric_name)

        for cs_stat in cumsum_metrics:
            ep_step_stats[cs_stat] = np.cumsum(ep_step_stats[cs_stat])

        return ep_step_stats

    def _get_ep_action_stats(self) -> (Tensor, Tensor):
        """
        Computes episode-wide action statistics, such as oscillation ratio and next hop spread.
        """
        LOG_CONSOLE_FUNCTION()

        N_over_time = [obs.get_birdseye_obs_graph().num_nodes for obs in self._observations]
        if len(set(N_over_time)) > 1:
            raise NotImplementedError("Can't yet calculate action stats for networks with "
                                      "varying number of nodes over time")
        N = N_over_time[0]
        if N < 2:
            raise ValueError("Can't calculate action stats for networks with less than 2 nodes")
        T = len(self._actions)
        next_hops = -torch.ones(N, N, T, dtype=torch.long)
        node_degrees = torch.zeros(N, T, dtype=torch.long)
        for t, actions_t in enumerate(self._actions):
            edge_src_t = actions_t.edge_index[0]
            node_degrees[:, t] = degree(edge_src_t, num_nodes=N)
            _, next_hops_t = scatter_max(actions_t.edge_attr, dim=0, index=edge_src_t, dim_size=N)  # shape: (N, N)
            next_hops[:, :, t] = next_hops_t

        # oscillation ratio: 0 when routing next_hops never change in between timesteps,
        # 1 when all next_hops change every timestep.
        next_hop_switches = torch.abs(next_hops[:, :, 1:] - next_hops[:, :, :-1])
        next_hop_switch_counts = (next_hop_switches != 0).sum(dim=-1)  # shape: [N, N]
        oscillation_ratio = next_hop_switch_counts.float() / (T - 1)  # shape: [N, N], per edge.src - flow.dst pair

        # next_hop spread: 0 when routing nodes use the same next_hop for all destinations,
        # 1 when next_hops follow a uniform spread per routing node
        next_hop_probs = scatter_add(torch.ones_like(next_hops), index=next_hops, dim=1).float() / N  # shape: [N, E, T]
        next_hop_logits = torch.clamp(torch.log(next_hop_probs), min=torch.finfo(next_hop_probs.dtype).min)
        next_hop_entropy = -(next_hop_probs * next_hop_logits).sum(dim=1)  # shape: [N, T]
        next_hop_spread = 1 - (next_hop_entropy / np.log(N)).mean(dim=1)  # shape: [N], per edge.src node

        return oscillation_ratio, next_hop_spread

    def _episode_to_hash_str(self) -> str:
        """
        Returns a hash string that uniquely identifies the episode trajectory for debugging purposes.
        For example, if you changed a detail that shouldn't change the trajectory, the hash should stay the same.
        The hash includes the following episode data:
        - all observations (raw),
        - all traffic matrices,
        - all actions,
        - all rewards.
        """
        LOG_CONSOLE_FUNCTION()

        parts = []

        for obs in self._observations:
            parts.append(obs.to_canonical_bytes())

        for tm in self._traffic_matrices:  # tm is a TrafficMatrix object (np.ndarray)
            parts.append(canonical_bytes(tm))

        for action in self._actions:  # action is a Data object, which contains edge_index, edge_attr, and agent_ids
            parts.append(action.to_canonical_bytes())

        for (reward_values, reward_observer_idx) in self._rewards:  # both are torch tensors
            parts.append(canonical_bytes(reward_values))
            parts.append(canonical_bytes(reward_observer_idx))

        return canonical_hash(parts)
