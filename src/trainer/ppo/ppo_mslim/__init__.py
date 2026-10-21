from omegaconf import DictConfig

from ._ppo_ms import PPOMSlimTrainer
from .ppo_ms_new import PPOMSlimNewTrainer
from .ppo_ms_path import PPOMSlimPathTrainer
from .ppo_ms_tmlr24 import PPOMSlimTMLR24Trainer
from src.algorithm.trainable.mslim import MSlimAlgorithm
from src.packerl_env import PackerlEnv


def get_ppo_mslim_trainer(config: DictConfig, trainer_config: DictConfig, env: PackerlEnv, algorithm: MSlimAlgorithm) \
    -> PPOMSlimTrainer:
    if trainer_config.exploration_mechanism == "new":
        return PPOMSlimNewTrainer(config, trainer_config, env, algorithm)
    elif trainer_config.exploration_mechanism == "path":
        return PPOMSlimPathTrainer(config, trainer_config, env, algorithm)
    elif trainer_config.exploration_mechanism == "tmlr24":
        return PPOMSlimTMLR24Trainer(config, trainer_config, env, algorithm)
    else:
        raise ValueError(f"Unknown exploration mechanism: {trainer_config.exploration_mechanism}")
