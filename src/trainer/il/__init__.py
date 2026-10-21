from omegaconf import DictConfig, open_dict

from ._il import ImitationLearningTrainer
from .il_fieldlines import ILFieldLinesTrainer
from .il_mslim import ILMSlimTrainer
from .il_placer import ILPlacerTrainer
from src.algorithm.trainable import TrainableAlgorithm
from src.algorithm.trainable.fieldlines import FieldlinesAlgorithm
from src.algorithm.trainable.mslim import MSlimAlgorithm
from src.algorithm.trainable.placer import PlacerAlgorithm
from src.packerl_env import PackerlEnv


def get_il_trainer(config: DictConfig, trainer_config: DictConfig, env: PackerlEnv, algorithm: TrainableAlgorithm) \
        -> ImitationLearningTrainer:

    with open_dict(trainer_config):
        il_trainer_config = trainer_config.copy()
        if isinstance(algorithm, FieldlinesAlgorithm):
            trainer = ILFieldLinesTrainer(config, il_trainer_config, env, algorithm)
        elif isinstance(algorithm, MSlimAlgorithm):
            il_trainer_config.merge_with(il_trainer_config.pop('mslim'))
            trainer = ILMSlimTrainer(config, il_trainer_config, env, algorithm)
        elif isinstance(algorithm, PlacerAlgorithm):
            il_trainer_config.merge_with(il_trainer_config.pop('placer'))
            trainer = ILPlacerTrainer(config, il_trainer_config, env, algorithm)
        else:
            raise NotImplementedError(f"IL trainer for algorithm {type(algorithm)} not implemented")
    return trainer
