from typing import Any, Dict, List

import torch
from torch import Tensor

from src.packerl_env import PackerlEnv
from src.scenario import Generator as ScenarioGenerator, NetworkScenario
from src.utils.logging import LOG_CONSOLE_RUN, RECORD_EVAL
from src.utils.observations import Observation, ObservationGraph, ObservationBatch
from src.utils.time import TimedBlock
from src.utils.types import ActionRaw, ActionSedi, LogprobTensor, ValueTensor, InferenceTimeTensor, Reward
from src.utils.utils import list_of_dict_to_dict_of_lists


class Algorithm:
    """
    Base class for all routing algorithms usable in PackeRL.
    """
    action_raw_representation: str  # specifies what the action_raw tensor represents ("next_hop", "link_weight" or "node_emb")

    # set during algorithm creation in src.algorithm.__init__.py
    _vali_scenario_generator: ScenarioGenerator  # scenario generator to use during validation
    _eval_scenario_generator: ScenarioGenerator  # scenario generator to use during evaluation
    _eval_actor_loc: str  # actor location to use during evaluation ("single" or "multi")
    _eval_actor_input: str  # actor input representation to use during evaluation ("central", "local", "birdseye")

    def __init__(self, env: PackerlEnv):
        """
        Initializes the algorithm with the given parameters. An algorithm requires an intact PackeRL environment.
        By default, the algorithm is not trainable, but this can be set to True in the subclass constructor.
        """
        self._env = env

        # current deployment (location, input representation) - is set and un-set in the env interaction loop
        self.cur_loc = None
        self.cur_input = None

    def set_loc(self, actor_loc: str, actor_input: str):
        """
        Sets the current deployment (location, input representation) of the algorithm.
        This is used to inform the algorithm about the current deployment during environment interaction.
        """
        self.cur_loc = actor_loc
        self.cur_input = actor_input

    # ====== Save & Load =========================================================

    def load(self, load_dir: str):
        """
        Load the algorithm's model from a file. This is used to load the model after training or to load a
        pre-trained model for evaluation.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    # ====== Env interaction =====================================================

    def reset_env(self, scenario: NetworkScenario) -> Observation:
        """
        The algorithm's environment reset method for evaluation. By default, it simply calls env.reset().
        """
        return self._prepare_obs(self._env.reset(scenario))

    def _prepare_obs(self, obs_raw: Observation) -> Observation | ObservationGraph:
        """
        Get corresponding view and filtering of obs
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_actor_view(self, obs: Observation | ObservationBatch) -> Observation | ObservationBatch:
        """
        TODO desc
        """
        return obs.get_actor_input(actor_loc=self.cur_loc, actor_input=self.cur_input)

    def _get_action(self, obs_graph: ObservationGraph) -> (ActionRaw, ActionSedi):
        """
        The algorithm's actual action selection method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_action(self, obs: Observation) -> (ActionRaw, ActionSedi, LogprobTensor, ValueTensor, InferenceTimeTensor):
        """
        Obtains the action for the given observations per agent, depending on the concrete algorithm implementation.
        Includes inference time measurement per agent.
        """
        obs_view_graphs = self.get_actor_view(obs).to_obs_graph_list()
        action_raw_list, action_sedi_list, inference_times_ms = [], [], []
        for obs_view in obs_view_graphs:
            with TimedBlock() as inference_timer:
                ac_raw, ac_sedi = self._get_action(obs_view)
            action_raw_list.append(ac_raw)
            action_sedi_list.append(ac_sedi)
            inference_times_ms.append(round(inference_timer.elapsed_ms, 3))

        return (torch.cat(action_raw_list),
                torch.cat(action_sedi_list),
                torch.tensor(0.0),  # logprob (not used in eval)
                torch.tensor(0.0),  # value (not used in eval)
                torch.tensor(inference_times_ms, dtype=torch.float32)
                )

    def get_action_sedi_offsets(self, obs_view: Observation) -> Tensor:
        N_per_graph = obs_view.num_nodes_per_graph
        E_per_graph = obs_view.num_edges_per_graph
        sedi_entries_per_graph = N_per_graph ** 2

        if self.cur_loc == "single":
            # sedi are provided without offset, so no offsets are needed
            sedi_offsets = torch.zeros(torch.sum(sedi_entries_per_graph), dtype=torch.long)
        else:  # self.cur_loc == "multi"
            # sedi are provided with offset, so we need to calculate the offsets to de-offset the sedi later
            max_sedi_value_per_graph = N_per_graph * E_per_graph
            sedi_offsets_per_graph = torch.cat((
                torch.tensor([0], dtype=torch.long),
                torch.cumsum(max_sedi_value_per_graph, dim=0)[:-1]
            ))
            sedi_offsets = torch.repeat_interleave(sedi_offsets_per_graph, sedi_entries_per_graph)

        return sedi_offsets

    def get_action_sedi_mask(self, obs_view: Observation | ObservationBatch) -> Tensor:
        N_per_graph = obs_view.num_nodes_per_graph
        observer_ids_per_agent_id = obs_view.observer_ids.repeat_interleave(N_per_graph)

        if self.cur_loc == "single":
            # The action stems from a single observation (graph) -> all entries are valid
            agent_id_mask = torch.ones_like(observer_ids_per_agent_id, dtype=torch.bool)

        else:  # self.cur_loc == "multi"
            # The action stems from per-agent observation graphs -> we take only those sedi values of the action
            # where the agent ID of respective edge source node corresponds to the graph observer
            agent_id_mask: Tensor = observer_ids_per_agent_id == obs_view.agent_ids

        # the sedi mask covers N^2 values; one per (routing node, destination node) pair per graph
        return agent_id_mask.repeat_interleave(N_per_graph.repeat_interleave(N_per_graph))

    def post_step(self,
                  obs: Observation,
                  action_raw: ActionRaw,
                  action_sedi: ActionSedi,
                  logprob: LogprobTensor,
                  value: ValueTensor,
                  reward: Reward,
                  next_obs_raw: Observation,
                  done: bool,
                  infos: Dict[str, Any]) -> Observation:
        """
        The algorithm's post-processing method for evaluation.
        It can be used to normalize rewards, handle buffers, etc.
        """
        next_obs = self._prepare_obs(next_obs_raw)  # filter out features, add auxiliary features, etc.
        return next_obs

    # ====== Env interaction loop for training, validation, and evaluation =======

    def _compute_agent_delays(self,
                              obs: Observation,
                              inference_times_ms: Tensor,
                              inf_delay_scaling: float) -> Tensor:
        """
        Computes the agent delays tensor based on the observation and the current deployment location.
        These delays mimic the time it takes for each agent to receive and install its action after inference.
        """
        agent_ids = obs.get_birdseye_obs_graph().agent_ids  # Use the 'actual' network topology for delay computation

        if self._env.episode_has_just_begun or self.cur_input == "birdseye":
            # No delays when at the beginning of an episode (A.), or when using the 'birdseye' input (B.):
            # A. Conceptually, we assume that all agents enter the episode with an available 'base action';
            #    Technically, we need fully populated routing tables in the ns-3 sim for the first step.
            # B. Using the 'birdseye' input means we disregard communication delays (i.e. this is a 'toy' setting).
            return torch.zeros_like(agent_ids, dtype=torch.float32)
        else:
            # In all other cases, overall agent delay consists of communication delay + inference time.
            # 1. Communication delay depends on the input type
            if self.cur_input == "central":
                # Using the 'central' input means that we have assembled the observations to a global view
                # in a central location, and we are doing inference either centrally or in a distributed manner.
                # In both cases, we'd need to re-distribute information to the agents.
                dests, delays = self._env.comm_delays_from_central_node
                perm = torch.argsort(dests)  # re-order delays to match agent order in observation
                pos = torch.searchsorted(dests[perm], agent_ids, right=False)
                comm_delays_to_agents = delays[perm[pos]]
            elif self.cur_input == "local":
                # Using 'local' input means that we have already transported the observations to the agents and
                # inference happened there.
                comm_delays_to_agents = torch.zeros_like(agent_ids, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown actor_input: {self.cur_input}")

            # 2. Inference times depend on the actor location
            if self.cur_loc == "single":
                # Central node inferred action for all agents - all agents have the same central agent delay
                assert inference_times_ms.numel() == 1, "Expected a single inference time for 'single' actor location."
                inference_delays = inference_times_ms.expand_as(agent_ids)
            elif self.cur_loc == "multi":
                # Each agent inferred its own action - each agent has its own agent delay
                assert inference_times_ms.numel() == agent_ids.numel(), \
                    "Expected per-agent inference times for 'multi' actor location."
                inference_delays = inference_times_ms
            else:
                raise ValueError(f"Unknown actor_loc: {self.cur_loc}")

            # overall delays in installing actions: communication delays + scaled inference delays
            return comm_delays_to_agents + inf_delay_scaling * inference_delays

    def run_environment(self,
                        env_mode: str,
                        scenarios: List[NetworkScenario],
                        actor_loc: str,
                        actor_input: str,
                        reset_func,
                        get_action_func,
                        post_step_func) -> List[Dict[str, Any]]:
        """
        Runs the specified number of steps (num_episodes x ep_length) in the environment,
         and collects the results in a list of dictionaries (one per episode).
         For the first `visualize_first_n_episodes` episodes, visualizations are created.

        Returns: a list of dictionaries, each containing the results of an episode incl. per-step logs.
        """
        num_episodes = len(scenarios)
        total_steps = sum(s.length for s in scenarios)
        LOG_CONSOLE_RUN(f"--- {env_mode} ({num_episodes=}, {total_steps=}, deployment: {(actor_loc, actor_input)}) ---")

        ep_logs = []  # list of episode summary dictionaries
        step_logs = []  # list of step log dictionaries for the current episode

        self.set_loc(actor_loc, actor_input)  # inform algorithm about current deployment
        if env_mode in ["validation", "evaluation"]:
            inf_delay_scaling = self._env.config.eval_inference_delay_scaling
        else:
            inf_delay_scaling = self._env.config.train_inference_delay_scaling
        next_scenario = scenarios.pop(0)
        obs = reset_func(next_scenario)

        for i in range(total_steps):

            # reset environment if not running
            if not self._env.in_episode:
                next_scenario = scenarios.pop(0)
                obs = reset_func(next_scenario)

            # get action
            action_raw, action_sedi, logprob, value, inference_times_ms = get_action_func(obs)

            # prepare action for environment (combine potential multi-agent sedi into a single tensor) and execute step
            with torch.no_grad():
                obs_view = self.get_actor_view(obs)
                sedi_mask = self.get_action_sedi_mask(obs_view).to(action_sedi.device)
                action_sedi_for_env = action_sedi[sedi_mask].detach().cpu()
                agent_delays = self._compute_agent_delays(obs, inference_times_ms, inf_delay_scaling).detach().cpu()
                if (self.action_raw_representation == "node_emb"
                        and getattr(self, "embedding_dim", None) == 2
                        and self.cur_loc == "single"
                        and env_mode in ["validation", "evaluation"]):
                    node_emb = action_raw.detach().cpu()
                else:
                    node_emb = None
                next_obs_raw, reward, done, infos = self._env.step(action_sedi_for_env, agent_delays, node_emb)

            # post-step processing
            # (e.g. obs preparation, obs/reward normalization, supervised loss calc., buffer filling, etc.)
            with TimedBlock() as post_step_timer:
                next_obs = post_step_func(obs, action_raw, action_sedi, logprob, value, reward,
                                          next_obs_raw, done, infos)

            # log step results / episode summary
            step_log = infos | {"inference_time_ms": inference_times_ms.mean().item(),
                                "post_step_time_ms": round(post_step_timer.elapsed_ms, 3),
                                }
            if done:
                ep_summary = step_log.pop("ep_summary")  # ep_summary is in step_log of last step of episode
                step_logs.append(step_log)
                ep_summary |= list_of_dict_to_dict_of_lists(step_logs)
                ep_logs.append(ep_summary)
                step_logs = []
            else:
                step_logs.append(step_log)

            obs = next_obs

        return ep_logs

    # ============================================================================

    def validate(self, iteration_log: dict):
        with TimedBlock() as validation_timer:
            self._vali_scenario_generator.reset_rng()  # reset RNG to ensure we always validate on the same scenarios
            validation_scenarios = self._vali_scenario_generator.generate()
            validation_summaries = self.run_environment(env_mode="validation",
                                                        scenarios=validation_scenarios,
                                                        actor_loc=self._eval_actor_loc,
                                                        actor_input=self._eval_actor_input,
                                                        reset_func=self.reset_env,
                                                        get_action_func=self.get_action,
                                                        post_step_func=self.post_step
                                                        )
        iteration_log["validation"] = validation_summaries
        iteration_log["validation_time_s"] = round(validation_timer.elapsed_s, 3)

    def evaluate(self):
        LOG_CONSOLE_RUN("")
        LOG_CONSOLE_RUN(f"=== evaluation ===")
        with TimedBlock() as eval_timer:
            self._eval_scenario_generator.reset_rng()  # reset RNG to ensure we always evaluate on the same scenarios
            evaluation_episodes = self._eval_scenario_generator.generate()
            evaluation_summaries = self.run_environment(env_mode="evaluation",
                                                        scenarios=evaluation_episodes,
                                                        actor_loc=self._eval_actor_loc,
                                                        actor_input=self._eval_actor_input,
                                                        reset_func=self.reset_env,
                                                        get_action_func=self.get_action,
                                                        post_step_func=self.post_step
                                                        )
        LOG_CONSOLE_RUN(f"--- saving records ---")
        RECORD_EVAL(evaluation_summaries, eval_timer.elapsed_s)
