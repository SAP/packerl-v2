import torch
from torch import Tensor

from .._ppo import PPOTrainer

from src.utils.observations import Observation, ObservationBatch
from src.utils.observations.obs_utils import get_nexthop_candidate_idx
from src.utils.types import ActionRaw, ActionSedi, LogprobTensor, EntropyTensor


class PPOFieldlinesTrainer(PPOTrainer):

    def _get_value_func_feat_counts(self, feat_counts: dict):
        return feat_counts

    def _get_value_view(self, obs: Observation) -> Observation:
        return obs.get_value_input(value_loc=self.train_value_loc,
                                   value_input=self.train_value_input,
                                   link_weights_as_input=False)

    def _sample_edges(self, edge_dest_values, edge_dest_idx) -> (ActionRaw, ActionSedi):
        """
        Given the actor output per edge per destination and an index tensor, this method actually selects the
        edge actions. The derived classes need to implement this method, e.g. by sampling or selecting the max.
        """
        raise NotImplementedError

    def _compare_actions(self,
                         action_raw: ActionRaw,
                         nci: Tensor,
                         other_action_raw: ActionRaw,
                         other_action_sedi: ActionSedi) -> (LogprobTensor, EntropyTensor, EntropyTensor):
        raise NotImplementedError

    def _evaluate_action(self,
                         rollout_obs_batch_view: ObservationBatch,
                         rollout_action_raw: ActionRaw,
                         rollout_action_sedi: ActionSedi) -> (LogprobTensor, EntropyTensor, EntropyTensor):
        """
        Evaluate the given action with respect to the actor's output from the given observation batch.
        """
        # get unrolled edge-dest indices to properly index the edge_dest_actor_output tensor
        #  Caution: this works on PyG's inner node IDs (not PackeRL's agent IDs),
        #  and assumes that every node has at least one outgoing edge!
        nci = get_nexthop_candidate_idx(rollout_obs_batch_view)

        # get edge_dest_actor_output from the actor, which are interpreted depending on the sampling mode
        eval_action_raw = self._algo.actor(rollout_obs_batch_view.clone().to(self._algo.actor.device))  # 1-dim tensor, numel=sum([g.E*g.N for graph g in batch])

        return self._compare_actions(eval_action_raw, nci, rollout_action_raw, rollout_action_sedi)

    def _get_logprob_mask(self, rollout_obs_batch_view: ObservationBatch):
        observer_ids = rollout_obs_batch_view.observer_ids
        if self.train_actor_loc == "single":
            # The logprobs stem from a single observation graph per obs in the batch -> all entries are valid
            full_mask = torch.ones_like(observer_ids, dtype=torch.bool)
            return full_mask.repeat_interleave(rollout_obs_batch_view.num_nodes_per_graph ** 2)
        else:
            # The logprobs stem from per-agent observation graphs -> for every observation,
            #  we take only those logprob values of each observation graph that correspond to the graph observer
            N_per_graph = rollout_obs_batch_view.num_nodes_per_graph
            observer_ids_per_logprob_entry = observer_ids.repeat_interleave(N_per_graph ** 2)
            N_per_node = N_per_graph.repeat_interleave(N_per_graph)
            agent_ids_per_logprob_entry = rollout_obs_batch_view.agent_ids.repeat_interleave(N_per_node)
            return observer_ids_per_logprob_entry == agent_ids_per_logprob_entry

    def _get_logprob_graph_idx(self, rollout_obs_batch_view: ObservationBatch):
        idx_range = torch.arange(rollout_obs_batch_view.num_graphs, device=rollout_obs_batch_view.batch.device)
        return idx_range.repeat_interleave(rollout_obs_batch_view.num_nodes_per_graph ** 2)
