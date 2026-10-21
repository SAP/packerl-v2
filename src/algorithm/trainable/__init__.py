from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ._trainable import TrainableAlgorithm
from .fieldlines import FieldlinesAlgorithm
from .mslim import get_mslim_algorithm
from .placer import PlacerAlgorithm


def get_trainable_algorithm(env,
                            trainable_config: DictConfig,
                            save_path: Path = None,
                            load_path: Path = None,
                            device: str = "cpu") -> TrainableAlgorithm:
    flavor = trainable_config.flavor
    if flavor == "fieldlines":
        return FieldlinesAlgorithm(env, trainable_config, save_path, load_path, device)
    elif flavor == "mslim":
        return get_mslim_algorithm(env, trainable_config, save_path, load_path, device)
    elif flavor == "placer":
        return PlacerAlgorithm(env, trainable_config, save_path, load_path, device)
    else:
        raise ValueError(f"Unknown trainable algorithm flavor: {flavor}")


def create_or_load_trainable_algorithm(config: DictConfig, env, load_dir: str, device: str = "cpu") \
        -> TrainableAlgorithm:
    save_path = Path(config.run_dir)
    if not save_path.exists():
        raise FileNotFoundError(f"could not find the specified save directory: {save_path}. "
                                f"It should have been created when setting up the run directory.")
    if load_dir != "":
        load_path = Path(config.run_dir.replace(config.out_base_dir, load_dir))
        if not load_path.exists():
            raise FileNotFoundError(f"Load path {load_path} does not exist.")
        trainable_config = OmegaConf.load(load_path / "trainable_config.yaml")
    else:
        load_path = None
        trainable_config = config.algorithm.trainable
    return get_trainable_algorithm(env, trainable_config, save_path, load_path, device)
