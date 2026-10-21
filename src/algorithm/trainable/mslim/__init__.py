from omegaconf import DictConfig
from pathlib import Path

from .mslim import (MSlimAlgorithm,
                    NewLogSpaceMSlimAlgorithm,
                    NewFlatSpaceMSlimAlgorithm,
                    TMLR24LogSpaceMSlimAlgorithm,
                    TMLR24FlatSpaceMSlimAlgorithm)


def get_mslim_algorithm(env, trainable_config: DictConfig, save_path: Path, load_path: Path = None,
                        device: str = "cpu") -> MSlimAlgorithm:
    pred_format = trainable_config.mslim.pred_format
    pred_space = trainable_config.mslim.pred_space

    if pred_format == "dual":
        if pred_space == "log":
            return NewLogSpaceMSlimAlgorithm(env, trainable_config, save_path, load_path, device)
        elif pred_space == "flat":
            return NewFlatSpaceMSlimAlgorithm(env, trainable_config, save_path, load_path, device)
        else:
            raise ValueError(f"Unknown pred_space: {pred_space}")
    elif pred_format == "mean":
        if pred_space == "log":
            return TMLR24LogSpaceMSlimAlgorithm(env, trainable_config, save_path, load_path, device)
        elif pred_space == "flat":
            return TMLR24FlatSpaceMSlimAlgorithm(env, trainable_config, save_path, load_path, device)
        else:
            raise ValueError(f"Unknown pred_space: {pred_space}")
    else:
        raise ValueError(f"Unknown pred_format: {pred_format}")
