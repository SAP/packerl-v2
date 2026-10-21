from omegaconf import DictConfig, open_dict

from ._ppo import PPOTrainer
from .ppo_fieldlines import get_ppo_fieldlines_trainer
from .ppo_mslim import get_ppo_mslim_trainer
from .ppo_placer import get_ppo_placer_trainer
from src.algorithm.trainable import TrainableAlgorithm
from src.algorithm.trainable.fieldlines import FieldlinesAlgorithm
from src.algorithm.trainable.mslim import MSlimAlgorithm
from src.algorithm.trainable.placer import PlacerAlgorithm
from src.packerl_env import PackerlEnv


def get_ppo_trainer(config: DictConfig, trainer_config: DictConfig, env: PackerlEnv, algorithm: TrainableAlgorithm)  \
    -> PPOTrainer:

    with open_dict(trainer_config):
        ppo_trainer_config = trainer_config.copy()
        if isinstance(algorithm, FieldlinesAlgorithm):
            ppo_trainer_config.merge_with(ppo_trainer_config.pop("fieldlines"))
            trainer = get_ppo_fieldlines_trainer(config, ppo_trainer_config, env, algorithm)
        elif isinstance(algorithm, MSlimAlgorithm):
            ppo_trainer_config.merge_with(ppo_trainer_config.pop("mslim"))
            trainer = get_ppo_mslim_trainer(config, ppo_trainer_config, env, algorithm)
        elif isinstance(algorithm, PlacerAlgorithm):
            ppo_trainer_config.merge_with(ppo_trainer_config.pop("placer"))
            trainer = get_ppo_placer_trainer(config, ppo_trainer_config, env, algorithm)
        else:
            raise NotImplementedError(f"PPO trainer for algorithm {type(algorithm)} not implemented")
    return trainer
