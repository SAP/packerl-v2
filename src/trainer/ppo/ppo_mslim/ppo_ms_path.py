import math

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_scatter import scatter_mean

from ._ppo_ms import PPOMSlimTrainer
from src.trainer.dtrpl_layer import DtrplLayer
from src.algorithm.trainable.mslim.mslim import MSlimAlgorithm, NewMSlimAlgorithm, TMLR24MSlimAlgorithm
from src.utils.logging import LOG_CONSOLE_FUNCTION, LOG_CONSOLE_RUN, LOG_CONSOLE_INFO, LOG_CONSOLE_DEBUG
from src.utils.observations import ObservationGraph, ObservationBatch
from src.utils.sp_calculator import sp_to_action_sedi, apsp_to_action_sedi
from src.utils.time import TimedBlock
from src.utils.torch import train_stat_dict
from src.utils.types import ActionRaw, ActionSedi, LogprobTensor, EntropyTensor
from src.utils.utils import list_of_dict_to_dict_of_lists


def path_to_edge_onehot(path: list, edge_agent_index: Tensor, num_edges) -> Tensor:
    """
    Converts a path (list of agent IDs) to a one-hot encoded edge representation that is one at the edges
    traversed by the path and zero elsewhere.
    """
    edge_onehot = torch.zeros(num_edges, device=edge_agent_index.device)
    if len(path) < 2:
        return edge_onehot  # empty path
    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]
        # find the edge index corresponding to (src, dst)
        edge_mask = (edge_agent_index[0] == src) & (edge_agent_index[1] == dst)
        edge_indices = torch.nonzero(edge_mask, as_tuple=False).squeeze(-1)
        if edge_indices.numel() > 0:
            edge_onehot[edge_indices] = 1.0
    return edge_onehot


def split_rollout_action_raw(batch: ObservationBatch, action_raw: ActionRaw, K: int):

    n = batch.num_nodes_per_graph
    e = batch.num_edges_per_graph
    n2 = n * n
    idx_lens  = n2
    val_lens  = n2 * (K * e)
    block_lens = val_lens + idx_lens

    block_ends   = block_lens.cumsum(0)
    block_starts = block_ends - block_lens

    pos = torch.arange(action_raw.shape[0], device=action_raw.device)
    g = torch.bucketize(pos, block_ends[:-1], right=True)  # graph id per element
    within = pos - block_starts[g]
    is_value = within < val_lens[g]

    return action_raw[is_value], action_raw[~is_value]


class PPOMSlimPathTrainer(PPOMSlimTrainer):
    """
    PPO trainer for M-Slim algorithms that explore in path space.
    """
    target_entropy_factor_per_node_pair: float  # target entropy scaling (target ent. is log(#valid_paths) * this)
    K_top: int  # Number of top paths to consider
    K_rand: int  # Number of random paths to consider (in addition to top-K)
    rand_perturbation_scaling: float  # Scaling for random perturbations added to path scores when sampling random paths
    path_tau: float  # Temperature for path sampling distributions
    use_trpl: bool  # Whether to use Discrete TRPL for policy updates (instead of vanilla PPO)
    trpl_target_kl: float  # Target KL divergence for early stopping of TRPL policy updates (if use_trpl is True)
    trpl_alpha: float  # Coefficient for TRPL projection loss (encouraging the new policy to stay close to the old policy in terms of KL divergence).
    trpl_ent_coeff: float

    def __init__(self, config, trainer_config, env, algorithm):
        super().__init__(config, trainer_config, env, algorithm)
        self.K = self.K_top + self.K_rand
        if self.use_trpl:
            if self.train_actor_loc != "single":
                raise NotImplementedError("TRPL is not implemented yet for multi-agent training.")
            self.trpl = DtrplLayer()
            self.trpl_logit_buffer = []
        else:
            self.trpl = None
            self.trpl_logit_buffer = None

    def _sample_paths_single_agent(self,
                                   obs_nx: nx.DiGraph,
                                   link_weights: Tensor,
                                   proto_link_weights: Tensor | None,
                                   edge_agent_index: Tensor,
                                   agent_ids: Tensor,
                                   src_agent_id: int) -> (dict, (dict, dict), LogprobTensor, Tensor | None):

        assert isinstance(self._algo, MSlimAlgorithm)

        N = agent_ids.shape[0]
        E = edge_agent_index.shape[1]
        k_paths_all_dest = torch.empty((N, self.K, E), dtype=torch.float, device=agent_ids.device)
        sampled_idx_list = []
        sampled_paths = {}
        sampled_path_costs = {}
        num_sampled_random_paths = 0
        logprobs = []
        trpl_logits = [] if self.trpl_logit_buffer is not None else None

        for i, dst_agent_id in enumerate(agent_ids.tolist()):

            # initialize top-K with empty paths
            paths_list = [[]] * self.K
            paths = torch.zeros((self.K, E), dtype=torch.float, device=agent_ids.device)

            if dst_agent_id == src_agent_id:
                # no path needed to self, zero cost
                sampled_idx = torch.tensor(0, device=agent_ids.device)
                sampled_path_cost = 0
                logprob = torch.tensor(0.0, device=agent_ids.device)
                if trpl_logits is not None:
                    trpl_logits.append(torch.zeros(self.K, device=agent_ids.device))

            else:
                # get K_top shortest paths (at least 1). Using Yen's algorithm with time complexity O(K*N^3)
                sp_generator = nx.shortest_simple_paths(obs_nx, source=src_agent_id, target=dst_agent_id, weight='weight')
                try:
                    for k in range(self.K_top):
                        path = next(sp_generator)
                        paths_list[k] = path
                        paths[k] = path_to_edge_onehot(path, edge_agent_index, E)
                except StopIteration:
                    pass

                # obtain up to K_rand random paths by
                #  - computing the shortest path on a randomly perturbed version of the
                #    original graph (random perturbation of edge weights)
                #  - ensuring that the sampled path is not already in the top-K list
                #    (if it is, we discard and resample, up to some max number of attempts)
                if proto_link_weights is not None:
                    for k in range(self.K_top, self.K):  # doesn't enter if K_rand = 0
                        max_attempts = 5
                        attempts = 0
                        while attempts < max_attempts:
                            cur_perturbation = torch.randn_like(link_weights) * self.rand_perturbation_scaling
                            rand_weights = self._algo.map_link_weights(link_weights + cur_perturbation)
                            rand_weights_zip = zip(edge_agent_index[0].tolist(),
                                                   edge_agent_index[1].tolist(),
                                                   rand_weights.tolist())
                            rand_weights_nx = {(u, v): float(w) for u, v, w in rand_weights_zip}
                            nx.set_edge_attributes(obs_nx, rand_weights_nx, name="rand_weight")
                            try:
                                path = next(nx.shortest_simple_paths(obs_nx, source=src_agent_id, target=dst_agent_id, weight='rand_weight'))
                                if path not in paths_list:
                                    paths_list[k] = path
                                    paths[k] = path_to_edge_onehot(path, edge_agent_index, E)
                                    num_sampled_random_paths += 1
                                    break
                            except StopIteration:
                                pass
                            attempts += 1

                # check which paths are valid (non-empty)
                num_edges_per_path = paths.sum(dim=-1)  # (n_i * n_i, K)
                valid_paths = num_edges_per_path > 0  # (n_i * n_i, K)
                if not valid_paths.any():
                    raise ValueError(f"No valid paths found from {src_agent_id} to {dst_agent_id} "
                                     f"in the observation graph.")

                # compute path costs (inf for empty/invalid/non-existing paths)
                path_costs = paths @ link_weights  # shape (K,)
                masked_costs = path_costs.masked_fill(~valid_paths, float('inf'))

                # compute path distribution
                min_cost = masked_costs.min().detach()  # scalar
                path_logits = -(path_costs - min_cost) / self.path_tau
                path_logits = path_logits.masked_fill(~valid_paths, float('-inf'))
                path_dist = torch.distributions.Categorical(logits=path_logits)  # shape (K,)
                if trpl_logits is not None:
                    trpl_logits.append(path_dist.logits.detach().cpu())

                sampled_idx = path_dist.sample()
                sampled_path_cost = path_costs[sampled_idx].item()
                logprob = path_dist.log_prob(sampled_idx)

            k_paths_all_dest[i] = paths
            sampled_idx_list.append(sampled_idx)
            sampled_paths[dst_agent_id] = paths_list[sampled_idx.item()]
            sampled_path_costs[dst_agent_id] = sampled_path_cost
            logprobs.append(logprob)

        sssp = sampled_path_costs, sampled_paths
        logprob = torch.stack(logprobs)  # shape (N,)
        if trpl_logits is not None:
            trpl_logits = torch.stack(trpl_logits, dim=0)  # shape (N, K)
        sampled_idx_tensor = torch.stack(sampled_idx_list)  # shape (N,)

        if self.K_rand > 0:
            random_sample_quota = num_sampled_random_paths / ((N-1) * self.K_rand)
        else:
            random_sample_quota = 0.0
        return k_paths_all_dest, sampled_idx_tensor, sssp, logprob, trpl_logits, random_sample_quota

    def _sample_action(self, obs_graph: ObservationGraph) -> (ActionRaw, ActionSedi, LogprobTensor):

        assert isinstance(self._algo, MSlimAlgorithm)
        actor_input = obs_graph.clone().to(self._algo.actor.device)
        if self._algo.use_line_graph:
            actor_input = actor_input.to_line_graph(self._algo.line_graph_node_to_edge_features)
        link_weights = self._algo._get_link_weights(actor_input)
        if self.K_rand > 0:
            # if sampling random paths, too, provide the untransformed link weights to the sampling function
            # (so that it can apply random perturbations to them)
            proto_link_weights = self._algo.inv_map_link_weights(link_weights)
        else:
            proto_link_weights = None

        # create a NetworkX graph from the observation graph and link weights
        obs_nx = nx.DiGraph()
        obs_nx.add_nodes_from(obs_graph.agent_ids.tolist())
        obs_nx.add_weighted_edges_from(zip(obs_graph.edge_agent_index[0].tolist(),
                                           obs_graph.edge_agent_index[1].tolist(),
                                           link_weights.tolist()))

        N = obs_graph.agent_ids.shape[0]
        E = obs_graph.edge_agent_index.shape[1]

        k_paths = torch.zeros((N, N, self.K, E), dtype=torch.float, device=obs_graph.agent_ids.device)
        sampled_path_idx = torch.zeros((N, N), dtype=torch.long, device=obs_graph.agent_ids.device)
        logprobs = torch.zeros((N, N), dtype=torch.float, device=obs_graph.agent_ids.device)
        if self.trpl_logit_buffer is not None:
            trpl_logits = torch.zeros((N, N, self.K), dtype=torch.float, device=obs_graph.agent_ids.device)

        # a single centralized/birdseye agent: sample paths for all agents
        if self._algo.cur_loc == "single":

            apsp = []
            random_sample_quotas = []
            for i, src_agent_id in enumerate(obs_graph.agent_ids.tolist()):
                k_paths_all_dests_i, sampled_idx_tensor_i, sampled_paths_i, logprob_i, trpl_logits_i, sample_quota_i \
                    = self._sample_paths_single_agent(obs_nx,
                                                      link_weights,
                                                      proto_link_weights,
                                                      obs_graph.edge_agent_index,
                                                      obs_graph.agent_ids,
                                                      src_agent_id)
                k_paths[i] = k_paths_all_dests_i
                sampled_path_idx[i] = sampled_idx_tensor_i
                apsp.append((src_agent_id, sampled_paths_i))
                random_sample_quotas.append(sample_quota_i)
                logprobs[i] = logprob_i
                if self.trpl_logit_buffer is not None:
                    trpl_logits[i] = trpl_logits_i
            action_sedi = apsp_to_action_sedi(apsp, obs_graph.agent_ids, obs_graph.edge_agent_index)
            random_sample_quota = np.mean(random_sample_quotas)

        # a local agent: sample path for the observing agent only
        else:  # self._algo.cur_loc == "multi"
            src_agent_id = obs_graph.observer_ids.item()
            i = (obs_graph.agent_ids == src_agent_id).nonzero(as_tuple=False).item()
            k_paths_i, sampled_idx_tensor_i, sssp, logprob_i, trpl_logits_i, random_sample_quota \
                = self._sample_paths_single_agent(obs_nx,
                                                  link_weights,
                                                  proto_link_weights,
                                                  obs_graph.edge_agent_index,
                                                  obs_graph.agent_ids,
                                                  src_agent_id)
            k_paths[i] = k_paths_i
            sampled_path_idx[i] = sampled_idx_tensor_i
            logprobs[i] = logprob_i
            if self.trpl_logit_buffer is not None:
                trpl_logits[i] = trpl_logits_i
            action_sedi = sp_to_action_sedi(sssp, src_agent_id, obs_graph.agent_ids,
                                            obs_graph.edge_agent_index, obs_graph.edge_index)

        action_raw = torch.cat((k_paths.view(-1), sampled_path_idx.view(-1).float()), dim=0)
        if self.trpl_logit_buffer is not None:
            self.trpl_logit_buffer.append(trpl_logits.view(-1, self.K))  # stored shape: (N*N, K)
        LOG_CONSOLE_DEBUG(f"Random sample quota in this action: {random_sample_quota:.4f}")
        return action_raw, action_sedi, logprobs.view(-1)

    def _get_n2_mask(self, rollout_obs_batch_view: ObservationBatch):
        observer_ids = rollout_obs_batch_view.observer_ids
        if self.train_actor_loc == "single":
            # The entries stem from a single observation graph per obs in the batch -> all entries are valid
            full_mask = torch.ones_like(observer_ids, dtype=torch.bool)
            return full_mask.repeat_interleave(rollout_obs_batch_view.num_nodes_per_graph ** 2)
        else:
            # The entries stem from per-agent observation graphs -> for every observation,
            #  we take only those logprob values of each observation graph that correspond to the graph observer
            N_per_graph = rollout_obs_batch_view.num_nodes_per_graph
            observer_ids_per_logprob_entry = observer_ids.repeat_interleave(N_per_graph ** 2)
            N_per_node = N_per_graph.repeat_interleave(N_per_graph)
            agent_ids_per_logprob_entry = rollout_obs_batch_view.agent_ids.repeat_interleave(N_per_node)
            return observer_ids_per_logprob_entry == agent_ids_per_logprob_entry

    def _get_logprob_mask(self, rollout_obs_batch_view: ObservationBatch):
        return self._get_n2_mask(rollout_obs_batch_view)

    def _get_logprob_graph_idx(self, rollout_obs_batch_view: ObservationBatch):
        idx_range = torch.arange(rollout_obs_batch_view.num_graphs, device=rollout_obs_batch_view.batch.device)
        return idx_range.repeat_interleave(rollout_obs_batch_view.num_nodes_per_graph ** 2)

    def _get_entropy_mask(self, rollout_obs_batch_view: ObservationBatch) -> Tensor:
        return self._get_n2_mask(rollout_obs_batch_view)

    def _evaluate_action_batched(self,
                                 rollout_obs_batch_view: ObservationBatch,
                                 rollout_action_raw: ActionRaw,
                                 _: ActionSedi) -> (LogprobTensor, EntropyTensor, EntropyTensor):
        """
        Evaluates the action with respect to the given observation batch.
        """
        assert isinstance(self._algo, MSlimAlgorithm)
        if self._algo.use_line_graph:
            actor_input = (rollout_obs_batch_view.clone().to(self._algo.actor.device)
                           .to_line_graph(self._algo.line_graph_node_to_edge_features))
        else:
            actor_input = rollout_obs_batch_view.clone().to(self._algo.actor.device)
        if isinstance(self._algo, TMLR24MSlimAlgorithm):
            link_weights = self._algo.actor(actor_input).squeeze(-1)
        elif isinstance(self._algo, NewMSlimAlgorithm):
            actor_output = self._algo.actor(actor_input)
            link_weights, _ = self._algo._split_actor_output(actor_output)
        else:
            raise ValueError("unknown algorithm type")

        # rollout_action_raw is a concat of B tensors of shape (N_i * N_i * K * E_i + N_i * N_i)
        rollout_topk_paths, rollout_selected_paths \
            = split_rollout_action_raw(rollout_obs_batch_view, rollout_action_raw, self.K)

        raise NotImplementedError("batched evaluation not yet implemented")

        # extract topk paths and selected path indices
        # TODO scatter_sum?
        rollout_num_valid_paths = NotImplementedError  # TODO obtain num_valid_paths from topk tensor, ensure correct device

        # TODO: scatter_sum over link_weights tensor expanded to match rollout_topk_paths shape?
        new_path_costs = rollout_topk_paths @ link_weights
        # TODO: min_cost (rollout tensors contain self-paths which have zero cost)
        if self._algo.pred_in_flat_space:
            path_probs = torch.softmax(-new_path_costs, dim=-1)
            path_dist = torch.distributions.Categorical(probs=path_probs)
        else:
            path_logits = -new_path_costs
            path_dist = torch.distributions.Categorical(logits=path_logits)

        logprob = path_dist.log_prob(rollout_selected_path_idx)
        entropy = path_dist.entropy()
        target_entropy = self.target_entropy_factor_per_node_pair * rollout_num_valid_paths * math.log(self.K)

        return logprob, entropy, target_entropy

    def _evaluate_action(self,
                         rollout_obs_batch_view: ObservationBatch,
                         rollout_action_raw: ActionRaw,
                         _: ActionSedi) -> (LogprobTensor, EntropyTensor, EntropyTensor):
        """
        Evaluates the action with respect to the given observation batch.
        """
        assert isinstance(self._algo, MSlimAlgorithm)
        if self._algo.use_line_graph:
            actor_input = (rollout_obs_batch_view.clone().to(self._algo.actor.device)
                           .to_line_graph(self._algo.line_graph_node_to_edge_features))
        else:
            actor_input = rollout_obs_batch_view.clone().to(self._algo.actor.device)
        link_weights = self._algo._get_link_weights(actor_input)

        # for now, we split our batch and do this individually per graph
        N = rollout_obs_batch_view.num_nodes_per_graph
        E = rollout_obs_batch_view.num_edges_per_graph
        idx_lens = N * N
        val_lens = N * N * self.K * E
        block_lens = val_lens + idx_lens

        agent_ids_split = torch.split(rollout_obs_batch_view.agent_ids, N.tolist(), dim=0)
        action_raw_split = torch.split(rollout_action_raw, block_lens.tolist(), dim=0)
        link_weights_split = torch.split(link_weights, E.tolist(), dim=0)

        logprobs = []
        entropies = []
        target_entropies = []

        zipp = zip(rollout_obs_batch_view.observer_ids.tolist(), agent_ids_split,
                   N.tolist(), E.tolist(),
                   action_raw_split, link_weights_split)
        for observer_id, agent_ids, n_i, e_i, action_raw_i, link_weights_i in zipp:

            n2 = n_i * n_i
            logprob_full = torch.zeros(n2, device=link_weights_i.device)
            entropy_full = torch.zeros(n2, device=link_weights_i.device)
            target_entropy_full = torch.zeros(n2, device=link_weights_i.device)

            # extract topk paths and selected path indices (each over all n2 pairs of nodes)
            rollout_topk_paths, rollout_selected_path_idx \
                = torch.split(action_raw_i, [n2 * self.K * e_i, n2], dim=0)
            paths = rollout_topk_paths.view(n2, self.K, e_i)
            sampled_idx = rollout_selected_path_idx.round().long()

            # evaluate per node pair, but only on the relevant entries (rest stays zero)
            src_ids = agent_ids.repeat_interleave(n_i)
            dst_ids = agent_ids.repeat(n_i)
            if self.train_actor_loc == "single":
                # single-agent case: take entries for all N * N node pairs except self-paths  (M = N * (N - 1) )
                sel = torch.nonzero(src_ids != dst_ids, as_tuple=False).squeeze(-1)
            else:
                # multi-agent case: take only entries for the current observer, and also exclude self-paths (M = N - 1)

                sel = torch.nonzero((src_ids != dst_ids) & (src_ids == observer_id), as_tuple=False).squeeze(-1)

            paths_sel = paths[sel]  # (M, K, e_i)
            sampled_idx_sel = sampled_idx[sel]  # (M, )

            num_edges_per_path = paths_sel.sum(dim=-1)  # (M, K)
            valid_paths = num_edges_per_path > 0
            if not valid_paths.any(dim=-1).all():
                raise RuntimeError("Found (s,t) with zero valid candidates in stored Top-K.")

            path_costs = (paths_sel @ link_weights_i)  # (M, K)
            masked_costs = path_costs.masked_fill(~valid_paths, float('inf'))

            # compute path distribution
            min_cost = masked_costs.min(dim=-1, keepdim=True).values.detach()  # (M, 1)
            path_logits = -(masked_costs - min_cost) / self.path_tau
            path_logits = path_logits.masked_fill(~valid_paths, float('-inf'))

            path_dist = torch.distributions.Categorical(logits=path_logits)
            logprob_full[sel] = path_dist.log_prob(sampled_idx_sel)
            entropy_full[sel] = path_dist.entropy()

            # target entropy scaling depends on number of valid paths per node pair
            num_valid = valid_paths.sum(-1)  # Shape: (M, )
            target_entropy_full[sel] = (self.target_entropy_factor_per_node_pair *
                                        torch.log(num_valid.clamp_min(1).float()))

            logprobs.append(logprob_full)
            entropies.append(entropy_full)
            target_entropies.append(target_entropy_full)

        logprob_all = torch.cat(logprobs, dim=0)
        entropy_all = torch.cat(entropies, dim=0)
        target_entropy_all = torch.cat(target_entropies, dim=0)

        return logprob_all, entropy_all, target_entropy_all

    def _evaluate_action_trpl(self,
                              rollout_obs_batch_view: ObservationBatch,
                              rollout_action_raw: ActionRaw,
                              _: ActionSedi,
                              rollout_logits: Tensor) -> (LogprobTensor, Tensor, Tensor):
        """
        Evaluates the action with respect to the given observation batch.
        """
        assert isinstance(self._algo, MSlimAlgorithm)
        if self._algo.use_line_graph:
            actor_input = (rollout_obs_batch_view.clone().to(self._algo.actor.device)
                           .to_line_graph(self._algo.line_graph_node_to_edge_features))
        else:
            actor_input = rollout_obs_batch_view.clone().to(self._algo.actor.device)
        link_weights = self._algo._get_link_weights(actor_input)

        # for now, we split our batch and do this individually per graph
        N = rollout_obs_batch_view.num_nodes_per_graph
        E = rollout_obs_batch_view.num_edges_per_graph
        idx_lens = N * N
        val_lens = N * N * self.K * E
        block_lens = val_lens + idx_lens

        agent_ids_split = torch.split(rollout_obs_batch_view.agent_ids, N.tolist(), dim=0)
        action_raw_split = torch.split(rollout_action_raw, block_lens.tolist(), dim=0)
        link_weights_split = torch.split(link_weights, E.tolist(), dim=0)
        rollout_logits_split = torch.split(rollout_logits, (N*N).tolist(), dim=0)

        logprobs_list = []
        unprojected_logits_list = []
        new_logits_list = []

        zipp = zip(rollout_obs_batch_view.observer_ids.tolist(), agent_ids_split,
                   N.tolist(), E.tolist(),
                   action_raw_split, link_weights_split, rollout_logits_split)
        for observer_id, agent_ids, n_i, e_i, action_raw_i, link_weights_i, rollout_logits_i in zipp:

            n2 = n_i * n_i
            logprob_full = torch.zeros(n2, device=link_weights_i.device)
            unprojected_logits_full = torch.zeros((n2, self.K), device=link_weights_i.device)
            new_logits_full = torch.zeros((n2, self.K), device=link_weights_i.device)

            # extract topk paths and selected path indices (each over all n2 pairs of nodes)
            rollout_topk_paths, rollout_selected_path_idx \
                = torch.split(action_raw_i, [n2 * self.K * e_i, n2], dim=0)
            paths = rollout_topk_paths.view(n2, self.K, e_i)
            sampled_idx = rollout_selected_path_idx.round().long()

            # evaluate per node pair, but only on the relevant entries (rest stays zero)
            src_ids = agent_ids.repeat_interleave(n_i)
            dst_ids = agent_ids.repeat(n_i)
            if self.train_actor_loc == "single":
                # single-agent case: take entries for all N * N node pairs except self-paths  (M = N * (N - 1) )
                sel = torch.nonzero(src_ids != dst_ids, as_tuple=False).squeeze(-1)
            else:
                # multi-agent case: take only entries for the current observer, and also exclude self-paths (M = N - 1)

                sel = torch.nonzero((src_ids != dst_ids) & (src_ids == observer_id), as_tuple=False).squeeze(-1)

            paths_sel = paths[sel]  # (M, K, e_i)
            sampled_idx_sel = sampled_idx[sel]  # (M, )

            num_edges_per_path = paths_sel.sum(dim=-1)  # (M, K)
            valid_paths = num_edges_per_path > 0
            if not valid_paths.any(dim=-1).all():
                raise RuntimeError("Found (s,t) with zero valid candidates in stored Top-K.")

            path_costs = (paths_sel @ link_weights_i)  # (M, K)
            masked_costs = path_costs.masked_fill(~valid_paths, float('inf'))

            # compute path distribution
            min_cost = masked_costs.min(dim=-1, keepdim=True).values.detach()  # (M, 1)
            path_logits = -(masked_costs - min_cost) / self.path_tau
            path_logits = path_logits.masked_fill(~valid_paths, float('-inf'))

            path_dist = torch.distributions.Categorical(logits=path_logits)
            unprojected_logits = path_dist.logits  # (M, K)
            new_logits, _ = self.trpl(unprojected_logits, rollout_logits_i[sel], bound=self.trpl_target_kl)

            unprojected_logits_full[sel] = unprojected_logits
            new_logits_full[sel] = new_logits
            logprob_full[sel] = torch.gather(new_logits, dim=-1, index=sampled_idx_sel.unsqueeze(-1)).squeeze(-1)  # (M, )

            # target entropy scaling depends on number of valid paths per node pair

            logprobs_list.append(logprob_full)
            unprojected_logits_list.append(unprojected_logits_full)
            new_logits_list.append(new_logits_full)

        return (torch.cat(logprobs_list, dim=0),
                torch.cat(unprojected_logits_list, dim=0),
                torch.cat(new_logits_list, dim=0))

    def _train_step_vf_meta(self, vf_meta_ep_split_idx: Tensor, vf_meta_step_split_idx: Tensor):
        raise NotImplementedError("value function meta-learning not supported for path-based PPO")

    def _update_pi_from_minibatch_trpl(self, minibatch, rollout_logits: Tensor):
        """
        Update the policy from a given minibatch.
        """
        LOG_CONSOLE_FUNCTION()

        # unpack minibatch
        (rollout_obs_batch,
         _,
         rollout_actions_raw,
         rollout_actions_sedi,
         rollout_logprobs,
         _,
         advantages,
         _) = minibatch

        # evaluate the actions taken during the rollout phase
        rollout_obs_batch_view = self._algo.get_actor_view(rollout_obs_batch)
        with TimedBlock() as eval_actions_timer:
            train_logprobs, train_logits_unprojected, train_logits_new \
                = self._evaluate_action_trpl(rollout_obs_batch_view,
                                             rollout_actions_raw,
                                             rollout_actions_sedi,
                                             rollout_logits)

        # Get logprobs, ratios from logprobs (pre-clamping to avoid numerical instability in the exp() operation),
        # normalized advantages, and relevant entropies.
        # Logprobs of training and rollout should be the same values at first epoch.
        # We need to mask the logprobs to only include the values that are actually relevant for the ratio calculation
        # (e.g. filtering out logprob components of agent actions that fall outside the agent's sphere of influence).
        with TimedBlock() as prep_ratios_adv_entropy_timer:
            logprob_mask = self._get_logprob_mask(rollout_obs_batch_view)
            ratio_idx = self._get_logprob_graph_idx(rollout_obs_batch_view)

            train_logprobs = train_logprobs[logprob_mask]
            rollout_logprobs = rollout_logprobs[logprob_mask]
            train_logits_unprojected = train_logits_unprojected[logprob_mask]
            train_logits_new = train_logits_new[logprob_mask]
            rollout_logits = rollout_logits[logprob_mask]
            # obtain the ratios of the logprobs, then average them per graph (i.e. per step and agent) via ratio_idx
            ratios_raw = torch.exp(torch.clamp(train_logprobs - rollout_logprobs, min=-10.0, max=10.0))
            ratios = scatter_mean(ratios_raw, ratio_idx[logprob_mask], dim=0)
            normalized_advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8))

        # caution: some logit entries may be inf if they don't correspond to valid paths -> mask out resulting NaN
        mask = torch.isfinite(train_logits_unprojected)
        assert mask.equal(torch.isfinite(train_logits_new)), \
            print("train_logits_unprojected and train_logits_new should have NaNs in the same positions")
        assert mask.equal(torch.isfinite(rollout_logits)), \
            print("train_logits_unprojected and rollout_logits should have NaNs in the same positions")

        # calculate losses (all done with double precision).
        with TimedBlock() as loss_timer:
            policy_loss = -(normalized_advantages * ratios).mean()

            p_old = torch.where(mask, torch.exp(train_logits_unprojected), torch.zeros_like(train_logits_unprojected))
            train_logits_diff = train_logits_new - train_logits_unprojected.detach()
            tl_diff_masked = torch.where(mask, train_logits_diff, torch.zeros_like(train_logits_diff))
            projections = (p_old * tl_diff_masked).sum(dim=-1)
            projection_loss = projections.mean()

            p = torch.where(mask, torch.exp(train_logits_new), torch.zeros_like(train_logits_new))
            log_p = torch.where(mask, train_logits_new, torch.zeros_like(train_logits_new))
            entropy_loss = -(p * log_p).sum(dim=-1).mean()

        # OPTIMIZATION STEP
        with TimedBlock() as opt_step_timer:
            loss = policy_loss + self.trpl_alpha * projection_loss - self.trpl_ent_coeff * entropy_loss
            unclipped_pi_grads, clipped_pi_grads = self._opt_step(self.policy_optimizer, loss)

        # --- Logging ---
        # compute approximate KL divergence, clip fractions, and normalized entropy
        with torch.no_grad():
            # caution: some logit entries may be inf if they don't correspond to valid paths -> mask out resulting NaN

            p_new = torch.where(mask, torch.exp(train_logits_new), torch.zeros_like(train_logits_new))
            logits_diff = train_logits_new - rollout_logits.detach()
            l_diff_masked = torch.where(p_new > 0, logits_diff, torch.zeros_like(logits_diff))
            policy_kl = (p_new * l_diff_masked).sum(dim=-1).mean()

            proj_fraction = (projections > 1e-8).float().mean().cpu()
            ratio_outliers = torch.abs(ratios.detach() - 1) > self.pi_clip_range
            policy_clip_fraction = ratio_outliers.float().mean()

        train_step_scalars = {
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "policy_kl": policy_kl.item(),
            "projection_fraction": proj_fraction.item(),
            "policy_clip_fraction": policy_clip_fraction.item(),
        }
        train_step_scalars |= train_stat_dict(train_logprobs, "logprob")
        train_step_scalars |= train_stat_dict(rollout_logprobs, "old_logprob")
        train_step_scalars |= train_stat_dict(advantages, "adv")
        train_step_scalars |= train_stat_dict(normalized_advantages, "normalized_adv")
        train_step_scalars |= train_stat_dict(ratios, "ratios")
        train_step_scalars |= train_stat_dict(unclipped_pi_grads, "unclipped_pi_gradient")
        train_step_scalars["unclipped_pi_gradient"] = unclipped_pi_grads.detach()
        train_step_scalars |= train_stat_dict(clipped_pi_grads, "clipped_pi_gradient")
        train_step_scalars["clipped_pi_gradient"] = clipped_pi_grads.detach()

        times_s = {
            "evaluate_actions_time_s": round(eval_actions_timer.elapsed_s, 3),
            "ratio_adv_entropy_time_s": round(prep_ratios_adv_entropy_timer.elapsed_s, 3),
            "pi_loss_calc_time_s": round(loss_timer.elapsed_s, 3),
            "pi_opt_step_time_s": round(opt_step_timer.elapsed_s, 3),
        }

        return train_step_scalars, times_s

    def _train_step_trpl(self):
        """
        TODO desc
        """
        LOG_CONSOLE_FUNCTION()
        LOG_CONSOLE_RUN(f"--- training step ({self.update_epochs} update epochs, TRPL) ---")

        old_actor_params = torch.cat([param.detach().flatten() for param in self._algo.actor_params])
        old_value_params = torch.cat([param.detach().flatten() for param in self._value_function.parameters()])

        # preparation
        assert self._buffer.filled, "Cannot perform training step because the on-policy buffer is not full!"
        assert self.trpl is not None, \
            "TRPL logit buffer not initialized. Ensure that use_trpl is set to True in the trainer config."
        assert len(self.trpl_logit_buffer) == self._buffer.buffer_size, \
            f"TRPL logit buffer size {len(self.trpl_logit_buffer)} != rollout buffer size {self._buffer.buffer_size}"

        terminal_obs = ObservationBatch.from_obs_list(self._buffer.terminal_observations)
        _, terminal_values = self._get_value(terminal_obs, self._value_function, grad=False, get_detached_list=True)
        self._buffer.bootstrap_rewards(terminal_values, self.discount_factor)
        self._buffer.compute_returns_and_advantages(self.gae_lambda, self.discount_factor)

        update_infos = []
        unclipped_pi_grads = []
        clipped_pi_grads = []
        update_times = []

        # --- training core ---
        for e in range(self.update_epochs):
            update_infos_epoch = []
            update_times_epoch = []

            # Do a complete pass on the rollout buffer, sampling random minibatches.
            for minibatch, mb_idx in self._buffer.minibatches(self.minibatch_size, self._algo.actor.device):
                trpl_logits_mb = (torch.cat([self.trpl_logit_buffer[i] for i in mb_idx], dim=0)
                                  .to(self._algo.actor.device).detach())
                pi_update_info, pi_update_times_s = self._update_pi_from_minibatch_trpl(minibatch, trpl_logits_mb)
                unclipped_pi_grads.append(pi_update_info.pop("unclipped_pi_gradient"))
                clipped_pi_grads.append(pi_update_info.pop("clipped_pi_gradient"))

                vf_update_info, vf_update_times_s = self._update_vf_from_minibatch(minibatch,
                                                                                   self._value_function,
                                                                                   self.value_optimizer)
                update_infos_epoch.append(pi_update_info | vf_update_info)
                update_times_epoch.append(pi_update_times_s | vf_update_times_s)

            if "policy_kl" in update_infos_epoch[0]:
                mean_kl = np.mean([info["policy_kl"] for info in update_infos_epoch])
            else:
                mean_kl = None

            if "projection_fraction" in update_infos_epoch[0]:
                proj_frac = np.mean([info["projection_fraction"] for info in update_infos_epoch])
            else:
                proj_frac = None

            if "policy_clip_fraction" in update_infos_epoch[0]:
                mean_clip_frac = np.mean([info["policy_clip_fraction"] for info in update_infos_epoch])
            else:
                mean_clip_frac = None

            update_infos.extend(update_infos_epoch)
            update_times.extend(update_times_epoch)
            LOG_CONSOLE_INFO(f"update epoch {e}: "
                             f"mean KL = {mean_kl.round(3) if mean_kl is not None else None}, "
                             f"mean clip fraction = {mean_clip_frac.round(3) if mean_clip_frac is not None else None}, "
                             f"mean projection fraction = {proj_frac.round(3) if proj_frac is not None else None}, "
                             )
        # --- training core end ---

        # logging
        train_scalars = {}

        # - pi/ent/vf train scalars
        update_infos = list_of_dict_to_dict_of_lists(update_infos)
        for k, v in update_infos.items():
            train_scalars |= train_stat_dict(v, k)
        train_scalars["unclipped_pi_grad_variance"] = torch.stack(unclipped_pi_grads, dim=0).var(dim=0).mean().item()
        train_scalars["clipped_pi_grad_variance"] = torch.stack(clipped_pi_grads, dim=0).var(dim=0).mean().item()
        new_actor_params = torch.cat([param.detach().flatten() for param in self._algo.actor_params])
        actor_param_diffs = torch.abs(old_actor_params - new_actor_params)
        train_scalars["mean_actor_weight_difference"] = actor_param_diffs.mean().item()
        train_scalars["max_actor_weight_difference"] = actor_param_diffs.max().item()
        new_value_params = torch.cat([param.detach().flatten() for param in self._value_function.parameters()])
        value_param_diffs = torch.abs(old_value_params - new_value_params)
        train_scalars["mean_value_weight_difference"] = value_param_diffs.mean().item()
        train_scalars["max_value_weight_difference"] = value_param_diffs.max().item()
        train_scalars["value_explained_variance"] = self._buffer.get_explained_variance()
        train_scalars["entropy_rho"] = self._entropy_rho.item()
        if hasattr(self, "epsilon"):  # include epsilon if implemented in epsilon-greedy subclass
            train_scalars["epsilon"] = self.epsilon

        # - training times
        update_times = list_of_dict_to_dict_of_lists(update_times)
        train_scalars |= {f"times/{k}": round(np.sum(v), 3) for k, v in update_times.items()}  # training times

        return train_scalars

    def _train_step_default(self):
        if self.trpl is None:
            return super()._train_step_default()
        else:
            return self._train_step_trpl()

    def _train_iteration(self, it: int):
        if self.trpl_logit_buffer is not None:
            self.trpl_logit_buffer.clear()
        return super()._train_iteration(it)

