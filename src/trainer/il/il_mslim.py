from omegaconf import DictConfig
from torch import Tensor

from ._il import ImitationLearningTrainer
from src.algorithm.trainable import TrainableAlgorithm
from src.packerl_env import PackerlEnv
from src.utils.observations import Observation


class ILMSlimTrainer(ImitationLearningTrainer):

    link_weight_mse_variant: str  # determines which variant of link weight MSE loss to use

    def __init__(self, config: DictConfig, trainer_config: DictConfig, env: PackerlEnv, algorithm: TrainableAlgorithm):
        super().__init__(config, trainer_config, env, algorithm)

        if self.expert.action_raw_representation == "next_hop":
            raise ValueError("we can't train an M-Slim student with a next-hop based expert")

        if self.link_weight_mse_variant == "simple":
            from ._supervised_losses import link_weight_mse_loss
            self._link_weight_mse = link_weight_mse_loss
        elif self.link_weight_mse_variant == "dual_norm":
            from ._supervised_losses import link_weight_mse_loss_dual_normalized
            self._link_weight_mse = link_weight_mse_loss_dual_normalized
        elif self.link_weight_mse_variant == "dual_norm_v2":
            from ._supervised_losses import link_weight_mse_loss_dual_normalized_v2
            self._link_weight_mse = link_weight_mse_loss_dual_normalized_v2
        else:
            raise ValueError(f"Unknown link weight MSE loss variant: {self.link_weight_mse_variant}")

    def _get_supervised_loss(self,
                             action_raw: Tensor,
                             action_sedi_sparse: Tensor,
                             expert_action_raw: Tensor,
                             expert_sedi_sparse: Tensor,
                             input_obs: Observation) -> Tensor:
        """
        MSE loss between predicted and target link weights.
        """
        return self._link_weight_mse(student_link_weights=action_raw,
                                     expert_link_weights=expert_action_raw.to(action_raw.device)
                                     )
