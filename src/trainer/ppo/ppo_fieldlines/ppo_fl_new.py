import torch
from torch import Tensor
from torch_scatter import scatter_max, scatter_add

from src.utils.observations import ObservationGraph, ObservationBatch
from src.utils.observations.obs_utils import get_nexthop_candidate_idx
from src.utils.torch import scatter_logsumexp
from src.utils.types import ActionRaw, ActionSedi, LogprobTensor, EntropyTensor
from ._ppo_fl import PPOFieldlinesTrainer


class NewPPOFieldlinesTrainer(PPOFieldlinesTrainer):

    entropy_nexthop_kappa : float  # target ent. scaling factor for next-hop selection, set in abstract trainer init()

    def _sample_from_dist(self, edge_dest_values, edge_dest_idx) -> (ActionRaw, ActionSedi):
        """
        Samples next-hop edges per destination node based on the given edge destination probabilities and indices.
        The probabilities are normalized such that they sum to 1 for each group (= pair of routing x destination node).
        :param edge_dest_values: Interpreted as selection probabilities for each edge per destination node.
        :param edge_dest_idx: The indices of the edges per destination node, which are used to group the probabilities.
        """
        # Vectorized categorical sampling per group using the Gumbel–Max trick:
        #   sample u ~ Uniform(0, 1) for each element,
        #   compute g = -log(-log(u)) (Gumbel noise),
        #   then compute score = log(prob) + g.
        # The argmax of these scores per group gives the desired sample.
        probs_double = edge_dest_values.double()
        u = torch.rand_like(probs_double)
        gumbel_noise = -torch.log(-torch.log(u))
        scores = torch.log(probs_double) + gumbel_noise

        # For each group (= pair of routing x destination node), compute the index of the maximum score.
        # scatter_max returns (max_values, argmax_indices) per group.
        src_dst_pair_count = int(edge_dest_idx.max().item()) + 1
        _, sampled_edge_dest_idx = scatter_max(scores, edge_dest_idx, dim=0, dim_size=src_dst_pair_count)
        sampled_probs = edge_dest_values[sampled_edge_dest_idx]

        return sampled_probs, sampled_edge_dest_idx

    def _sample_action(self, obs_graph: ObservationGraph) -> (ActionRaw, ActionSedi, LogprobTensor):
        """
        Sample a next-hop routing preference, then select next-hop actions based on the sampled preferences.
        """
        edge_dest_logits = self._algo.actor(obs_graph.clone().to(self._algo.actor.device))

        # policy provides unnormalized edge logits -> normalize and turn into edge probabilities
        nci = get_nexthop_candidate_idx(obs_graph)
        edge_dest_logits_logsumexp = scatter_logsumexp(edge_dest_logits, nci)[nci]
        edge_probs = torch.exp(edge_dest_logits - edge_dest_logits_logsumexp)

        # select edges based on the edge probabilities and the exploration mode
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
        action_raw_logsumexp = scatter_logsumexp(action_raw, nci)[nci]  # action_raw are edge_dest_logits
        edge_probs = torch.exp(action_raw - action_raw_logsumexp)
        logprob = torch.log(edge_probs)[other_action_sedi]  # 1-dim tensor, numel=sum([g.N**2 for graph g in batch])

        src_dst_pair_count = int(nci.max().item()) + 1

        # entropy = -sum (p * log(p)) for each src-dst pair
        p_log_p = edge_probs * torch.log(edge_probs.clamp_min(1e-12))
        entropy = -scatter_add(p_log_p, nci, dim=0, dim_size=src_dst_pair_count)

        # Target entropy for nudging exploration for each src-dst pair
        ones = torch.ones_like(nci, dtype=edge_probs.dtype)
        K_per_group = scatter_add(ones, nci, dim=0, dim_size=src_dst_pair_count)
        target_entropy = self.entropy_nexthop_kappa * torch.log(K_per_group.clamp_min(1.0))

        return logprob, entropy, target_entropy

    def _get_entropy_mask(self, rollout_obs_batch_view: ObservationBatch) -> Tensor:
        return self._algo.get_action_sedi_mask(rollout_obs_batch_view)


class NewPPOFieldlinesBoltzmannTrainer(NewPPOFieldlinesTrainer):

    def _sample_edges(self, edge_dest_probs, edge_dest_idx) -> (ActionRaw, ActionSedi):
        """
        :param edge_dest_probs: Interpreted as selection probabilities for each edge per destination node.
        """

        # Boltzmann sampling from the edge destination probabilities
        return self._sample_from_dist(edge_dest_probs, edge_dest_idx)


class NewPPOFieldlinesEpsilonGreedyTrainer(NewPPOFieldlinesTrainer):

    # epsilon-greedy parameters, initialized in __init__ of abstract trainer class via the provided trainer_config
    initial_epsilon : float
    epsilon_decay : float
    min_epsilon : float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = 1.0

    def _set_iteration(self, it: int):
        self.epsilon = max(self.min_epsilon, self.initial_epsilon * self.epsilon_decay ** it)

    def _sample_edges(self, edge_dest_probs, edge_dest_idx) -> (ActionRaw, ActionSedi):
        """
        We sample the next-hop edges based on an epsilon-greedy strategy: with a probability of epsilon,
        we sample uniformly from the available edges, and with a probability of (1-epsilon), we select the greedy action
        (i.e. the edge with the highest probability). We achieve this by first executing both sampling strategies, for
        the entire batch and then selecting the sampled probs and positions based on a mask created from epsilon.
        """
        # Sample uniformly from the available edges
        uniform_probs = torch.ones_like(edge_dest_probs)
        uniformly_sampled_probs, uniformly_sampled_positions = self._sample_from_dist(uniform_probs, edge_dest_idx)

        # Get the greedy action by selecting the max prob edge for each batch element
        greedy_probs, greedy_positions = scatter_max(edge_dest_probs, index=edge_dest_idx)

        # Sampled positions are chosen based on a mask that implements epsilon-greedy exploration.
        eps_greedy_mask = torch.rand_like(uniformly_sampled_probs) < self.epsilon
        sampled_positions = torch.where(eps_greedy_mask, uniformly_sampled_positions, greedy_positions)

        # Sampled probabilities are those corresponding to the sampled positions
        sampled_probs = edge_dest_probs[sampled_positions]

        return sampled_probs, sampled_positions
