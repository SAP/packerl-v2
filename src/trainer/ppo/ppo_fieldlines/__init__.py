from omegaconf import DictConfig

from ._ppo_fl import PPOFieldlinesTrainer
from .ppo_fl_new import NewPPOFieldlinesBoltzmannTrainer, NewPPOFieldlinesEpsilonGreedyTrainer
from .ppo_fl_tmlr24 import TMLR24PPOFieldlinesBoltzmannTrainer, TMLR24PPOFieldlinesEpsGreedyTrainer
from src.algorithm.trainable.fieldlines import FieldlinesAlgorithm
from src.packerl_env import PackerlEnv


def get_ppo_fieldlines_trainer(config: DictConfig, trainer_config: DictConfig, env: PackerlEnv, algorithm: FieldlinesAlgorithm) \
    -> PPOFieldlinesTrainer:
    exp_mechanism = trainer_config.exploration_mechanism
    style = trainer_config.style
    if style == "new":
        if exp_mechanism == "boltzmann":
            return NewPPOFieldlinesBoltzmannTrainer(config, trainer_config, env, algorithm)
        elif exp_mechanism == "eps_greedy":
            return NewPPOFieldlinesEpsilonGreedyTrainer(config, trainer_config, env, algorithm)
        else:
            raise ValueError(f"Unknown exploration mechanism: {exp_mechanism}")
    elif style == "tmlr24":
        if exp_mechanism == "boltzmann":
            return TMLR24PPOFieldlinesBoltzmannTrainer(config, trainer_config, env, algorithm)
        elif exp_mechanism == "eps_greedy":
            return TMLR24PPOFieldlinesEpsGreedyTrainer(config, trainer_config, env, algorithm)
        else:
            raise ValueError(f"Unknown exploration mechanism: {exp_mechanism}")
    else:
        raise ValueError(f"Unknown style: {style}")
