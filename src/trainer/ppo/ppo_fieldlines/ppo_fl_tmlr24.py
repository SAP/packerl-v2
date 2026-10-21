import torch
from torch import Tensor
from torch_scatter import scatter_max, scatter_add
from torch.distributions.categorical import Categorical

from src.utils.observations import ObservationGraph, ObservationBatch
from src.utils.observations.obs_utils import get_nexthop_candidate_idx
from src.utils.torch import scatter_logsumexp
from src.utils.types import ActionRaw, ActionSedi, LogprobTensor, EntropyTensor
from ._ppo_fl import PPOFieldlinesTrainer


TMLR24_EXPLORATION_COEFF = -4.0  # Value taken from TMLR'24 paper


class TMLR24PPOFieldlinesTrainer(PPOFieldlinesTrainer):

    def _actor_output_to_probs(self, edge_dest_actor_output, edge_dest_idx) -> ActionRaw:
        """
        Converts the actor's output, i.e. edge selection logits, to a probability distribution over edges.
        Note that the resulting probabilities are similar but not identical
        to the softmax of the exponents of the logits (the usual way to obtain probs from logits), but we have to
        calculate them using the Log-Sum-Exp trick since otherwise we run into numerical overflows.
        """
        raise NotImplementedError

    def _sample_action(self, obs_graph: ObservationGraph) -> (ActionRaw, ActionSedi, LogprobTensor):
        """
        Sample a next-hop routing preference, then select next-hop actions based on the sampled preferences.
        """
        edge_dest_actor_output = self._algo.actor(obs_graph.clone().to(self._algo.actor.device))

        # turn actor output into probs, sample edges, and compute logprobs
        nci = get_nexthop_candidate_idx(obs_graph)
        edge_probs = self._actor_output_to_probs(edge_dest_actor_output, nci).double()
        selected_probs, action_sedi = self._sample_edges(edge_probs, nci)
        logprob = torch.log(selected_probs)  # 1-dim tensor, numel=sum([g.N**2 for graph g in batch])

        return edge_probs, action_sedi, logprob

    def _compare_actions(self,
                         action_raw: ActionRaw,
                         nci: Tensor,
                         other_action_raw: ActionRaw,
                         other_action_sedi: ActionSedi) -> (LogprobTensor, EntropyTensor, EntropyTensor):
        """
        Given the actor output per edge per destination, an index tensor and a previously computed (other) action,
        this method evaluates the given action wrt. its probability and entropy.
        """
        edge_probs = self._actor_output_to_probs(action_raw, nci).double()
        logprob = torch.log(edge_probs)[other_action_sedi]  # 1-dim tensor, numel=sum([g.N**2 for graph g in batch])

        # calculate entropy from categorical distribution over edges
        other_action_raw_double = other_action_raw.double()
        action_edge_p_logp = other_action_raw_double * torch.log(other_action_raw_double)
        entropy = -scatter_add(action_edge_p_logp, nci, dim_size=torch.numel(action_raw))  # 1-dim tensor, numel=sum([g.E*g.N for graph g in batch])
        target_entropy = None  # not used in TMLR'24 design

        return logprob, entropy, target_entropy

    def _get_entropy_mask(self, rollout_obs_batch_view: ObservationBatch) -> Tensor:
        E_per_graph = rollout_obs_batch_view.num_edges_per_graph
        observer_ids_per_edge = rollout_obs_batch_view.observer_ids.repeat_interleave(E_per_graph)

        if self.train_actor_loc == "single":
            # The action stems from a single observation (graph) -> all entries are valid
            agent_src_mask = torch.ones_like(observer_ids_per_edge, dtype=torch.bool)
        else:  # actor_loc == "multi"
            # The action stems from per-agent observation graphs -> we take only those raw values of the action
            # where the agent ID of respective edge source node corresponds to the graph observer
            agent_src_mask: Tensor = observer_ids_per_edge == rollout_obs_batch_view.edge_agent_index_without_offset[0]

        # modify mask shape based on policy design
        N_per_edge = rollout_obs_batch_view.num_nodes_per_graph.repeat_interleave(E_per_graph)
        return agent_src_mask.repeat_interleave(N_per_edge)

class TMLR24PPOFieldlinesBoltzmannTrainer(TMLR24PPOFieldlinesTrainer):
    def _actor_output_to_probs(self, edge_dest_actor_output, edge_dest_idx) -> ActionRaw:
        """
        Converts the actor's output, i.e. edge selection logits, to a probability distribution over edges.
        Note that the resulting probabilities are similar but not identical
        to the softmax of the exponents of the logits (the usual way to obtain probs from logits), but we have to
        calculate them using the Log-Sum-Exp trick since otherwise we run into numerical overflows.
        """
        edge_dest_logits = -TMLR24_EXPLORATION_COEFF * edge_dest_actor_output
        edge_dest_logits_logsumexp = scatter_logsumexp(edge_dest_logits, edge_dest_idx)
        return torch.exp(edge_dest_logits - edge_dest_logits_logsumexp[edge_dest_idx])

    def _sample_edges(self, edge_dest_values, edge_dest_idx) -> (ActionRaw, ActionSedi):
        """
        :param edge_dest_values: Interpreted as selection probabilities for each edge per destination node.
        """
        # We assume src_dst_pair_count corresponds to sum([N**2 for N in num_nodes_per_graph]),
        # which is the overall count of valid source-destination node pairs in the batch.
        src_dst_pair_count = int(edge_dest_idx.max().item()) + 1

        # Vectorized categorical sampling per group using the Gumbel–Max trick:
        #   sample u ~ Uniform(0, 1) for each element,
        #   compute g = -log(-log(u)) (Gumbel noise),
        #   then compute score = log(prob) + g.
        # The argmax of these scores per group gives the desired sample.
        probs_double = edge_dest_values.double()
        u = torch.rand_like(probs_double)
        gumbel_noise = -torch.log(-torch.log(u))
        scores = torch.log(probs_double) + gumbel_noise

        # For each group, compute the index of the maximum score.
        # scatter_max returns (max_values, argmax_indices) per group.
        _, sampled_edge_dest_idx = scatter_max(scores, edge_dest_idx, dim=0, dim_size=src_dst_pair_count)
        sampled_probs = edge_dest_values[sampled_edge_dest_idx]

        return sampled_probs, sampled_edge_dest_idx


class TMLR24PPOFieldlinesEpsGreedyTrainer(TMLR24PPOFieldlinesTrainer):

    # epsilon-greedy parameters, initialized in __init__ of abstract trainer class via the provided trainer_config
    epsilon_decay: float
    min_epsilon: float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = 1.0

    def _set_iteration(self, it: int):
        self.epsilon = self.min_epsilon + ((1 - self.min_epsilon) * self.epsilon_decay ** it)

    def _actor_output_to_probs(self, edge_dest_actor_output, edge_dest_idx) -> ActionRaw:
        """
        Converts the actor's output, i.e. edge selection logits, to a probability distribution over edges.
        Note that the resulting probabilities are similar but not identical
        to the softmax of the exponents of the logits (the usual way to obtain probs from logits), but we have to
        calculate them using the Log-Sum-Exp trick since otherwise we run into numerical overflows.
        """
        edge_dest_logits = -TMLR24_EXPLORATION_COEFF * edge_dest_actor_output
        edge_dest_logits_logsumexp = scatter_logsumexp(edge_dest_logits, edge_dest_idx)
        return torch.exp(edge_dest_logits - edge_dest_logits_logsumexp[edge_dest_idx])

    def _sample_edges(self, edge_dest_values, edge_dest_idx) -> (ActionRaw, ActionSedi):
        """
        edge_dest_scores are interpreted as unnormalized logits for a distribution, from which we sample the edges.
        The actor's output is scaled by the negative exploration coefficient C before being treated as a logit.
        However, with a probability of epsilon, we simply choose random edges.
        All in all, the input probs for the logprob operation are adjusted to account for eps.-greedy exploration.
        """
        # exploration: Sample random edges and set uniform probs
        if torch.rand(1) < self.epsilon:
            get_sample_from_probs = lambda cur_probs: Categorical(probs=torch.ones_like(cur_probs)).sample()
            get_factored_prob = lambda raw_prob, cur_options: self.epsilon / cur_options

        # greedy action: sample from categorical distribution spanned by probs and use according probs
        else:
            get_sample_from_probs = lambda cur_probs: Categorical(probs=cur_probs).sample()
            get_factored_prob = lambda raw_prob, cur_options: (1-self.epsilon) * raw_prob + self.epsilon / cur_options

        # We assume src_dst_pair_count corresponds to sum([N**2 for N in num_nodes_per_graph]),
        # which is the overall count of valid source-destination node pairs in the batch.
        src_dst_pair_count = int(edge_dest_idx.max().item()) + 1
        sampled_probs = torch.zeros(src_dst_pair_count)
        sampled_positions = torch.zeros(src_dst_pair_count, dtype=torch.long)

        # We don't have anything like scatter_sample() at the moment, so sampling is looped.
        for i in range(src_dst_pair_count):
            mask: torch.Tensor = edge_dest_idx == i
            cur_probs = edge_dest_values[mask]
            cur_options = cur_probs.nelement()
            if cur_options == 0:
                raise ValueError("there is no outgoing edge for some src-dst pair! This should not happen...")
            s = get_sample_from_probs(cur_probs)
            sampled_probs[i] = get_factored_prob(cur_probs[s], cur_options)
            cur_position_candidates = torch.nonzero(mask).view(-1)
            sampled_positions[i] = cur_position_candidates[s]

        return sampled_probs, sampled_positions
