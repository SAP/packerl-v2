from typing import Union, Optional, List

from omegaconf import DictConfig, open_dict

from .il import get_il_trainer
from .ppo import get_ppo_trainer
from ._trainer import Trainer
from src.algorithm.trainable import TrainableAlgorithm
from src.packerl_env import PackerlEnv
from src.utils.logging import LOG_CONSOLE_RUN


def is_trainer_name(name: str) -> bool:
    return name is not None and name not in ["none", "", "null"]


def get_trainer_names(trainer_names: Optional[Union[str, List[str]]]) -> list[str]:
    if isinstance(trainer_names, str):
        trainer_names = [name.strip() for name in trainer_names.split(",")]
    elif isinstance(trainer_names, list):
        trainer_names = []
        for t_name in trainer_names:
            trainer_names.extend(get_trainer_names(t_name))
    else:
        trainer_names = []
    trainer_names = [name for name in trainer_names if is_trainer_name(name)]
    return trainer_names


def get_trainer(trainer_name: str, config: DictConfig, env: PackerlEnv, algorithm: TrainableAlgorithm) -> Trainer:
    """
    Given a trainer name, config, environment, and algorithm, returns the corresponding trainer instance.
    """
    LOG_CONSOLE_RUN(f"=== initializing '{trainer_name}' trainer ===")

    with open_dict(config):
        trainer_config = config.trainer.copy()
        if trainer_name == "il":
            trainer_config.merge_with(trainer_config.pop("il"))
            trainer = get_il_trainer(config, trainer_config, env, algorithm)
        elif trainer_name == "ppo":
            trainer_config.merge_with(trainer_config.pop("ppo"))
            trainer = get_ppo_trainer(config, trainer_config, env, algorithm)
        else:
            raise ValueError(f"Unknown trainer name: {trainer_name}")
    return trainer
