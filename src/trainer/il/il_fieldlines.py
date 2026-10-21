from torch import Tensor

from ._il import ImitationLearningTrainer
from ._supervised_losses import sedi_sparse_ce_loss
from src.utils.observations import Observation


class ILFieldLinesTrainer(ImitationLearningTrainer):

    def _get_supervised_loss(self,
                             action_raw: Tensor,
                             action_sedi_sparse: Tensor,
                             expert_action_raw: Tensor,
                             expert_sedi_sparse: Tensor,
                             input_obs: Observation) -> Tensor:
        """
        Cross-entropy loss between FL's prediction edge-dest values and the sparse target sedi.

        pred_flat:   [E*N]   float logits, flattened so index = edge_idx * N + dest_idx
        target_flat: [E*N]   binary mask (0/1), flattened the same way
        input_obs:   Observation, must contain edge_index [2, E] tensor

        Returns:
          scalar: average cross-entropy over all (source s, dest j) pairs
        """
        return sedi_sparse_ce_loss(
            student_sedi_soft_sparse=action_raw.view(-1),
            expert_sedi_onehot_sparse=expert_sedi_sparse.to(action_raw.device),
            edge2src=input_obs.edge_index[0].to(action_raw.device)
        )
