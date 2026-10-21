from omegaconf import DictConfig

from .ppo_pl import PPOPlacerTrainer, PPOPlacerBoltzmannTrainer, PPOPlacerEpsilonGreedyTrainer
from src.algorithm.trainable.placer import PlacerAlgorithm
from src.packerl_env import PackerlEnv


def get_ppo_placer_trainer(config: DictConfig, trainer_config: DictConfig, env: PackerlEnv, algorithm: PlacerAlgorithm) \
    -> PPOPlacerTrainer:
    exp_mechanism = trainer_config.exploration_mechanism
    if exp_mechanism == "boltzmann":
        return PPOPlacerBoltzmannTrainer(config, trainer_config, env, algorithm)
    elif exp_mechanism == "eps_greedy":
        return PPOPlacerEpsilonGreedyTrainer(config, trainer_config, env, algorithm)
    else:
        raise ValueError(f"Unknown exploration mechanism: {exp_mechanism}")
