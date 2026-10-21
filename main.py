import random
import os
os.environ.setdefault(
    "NPY_DISABLE_CPU_FEATURES",
    "AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL",
)  # some machines have this, some don't. For full reproducibility of the scenario generation, disable it for now.

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

from src.algorithm import get_algorithm
from src.algorithm.trainable import TrainableAlgorithm
from src.packerl_env import PackerlEnv
from src.utils.config import pre_hydra, update_config_with_derived_values
from src.utils.logging import setup_console_logger, setup_experiment_recorder
from src.utils.torch import torch_seed_fork
from src.utils.utils import ensure_sysctl_value
from src.trainer import get_trainer_names, get_trainer


@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(config: DictConfig):

    # initialize, update, check configuration, and dump it to the newly created run directory
    with open_dict(config):
        update_config_with_derived_values(config)

    # if specified, print config to console and exit
    if config.only_print_config:
        print(OmegaConf.to_yaml(config, resolve=True))
        return

    import sys
    import traceback
    # This main is used to circumvent a bug in Hydra
    # See https://github.com/facebookresearch/hydra/issues/2664

    try:
        actual_main(config)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # flush everything
        sys.stdout.flush()
        sys.stderr.flush()


def actual_main(config: DictConfig):

    # Initializes the logging and experiment recorder for this environment.
    setup_console_logger(config.logging.log_level)
    setup_experiment_recorder(config)

    # algorithm seeding (does not affect scenario generation)
    random.seed(config.train_algo_seed)
    np.random.seed(config.train_algo_seed)
    torch.manual_seed(config.train_algo_seed)
    torch.cuda.manual_seed_all(config.train_algo_seed)

    # for C++ profiling, we use linux's 'perf' for profiling the ns3 code and simply change
    # the command to call ns3 via ns3-ai. This requires special privileges in
    # 'kernel.{kernel.perf_event_paranoid,kptr_restrict}'.
    if config.profiling_cpp:
        print("requested C++ profiling -> checking kernel permissions...")
        ensure_sysctl_value('kernel.perf_event_paranoid', -1)
        ensure_sysctl_value('kernel.kptr_restrict', 0)

    # for python profiling, we simply wrap the run statement in a cProfile object,
    # and extract stats after the run.
    if config.profiling_py:
        print("requested Python profiling -> running in cProfile mode...")
        import cProfile
        import pstats
        with cProfile.Profile() as profile:
            with PackerlEnv(config) as env:
                run(config, env)
            results = pstats.Stats(profile)
        stats_file = os.path.join(config.run_dir, "profiling_py.stats")
        print(f"writing profiling results to {stats_file}...")
        results.sort_stats(pstats.SortKey.TIME)
        results.dump_stats(stats_file)

    # if not using python profiling, we just start the run.
    else:
        with PackerlEnv(config) as env:
            run(config, env)


def run(config: DictConfig, env: PackerlEnv):
    """
    A run consists of training (if applicable) and evaluation.
    """
    with torch_seed_fork(config.module_init_seed):
        algorithm = get_algorithm(config, env, device=config.device)

    if isinstance(algorithm, TrainableAlgorithm) and not config.eval_only:
        trainer_names = get_trainer_names(config.trainer.trainers)
        if trainer_names:
            algorithm.save("initial save")
            total_its = 0

            for i, trainer_name in enumerate(trainer_names):

                with torch_seed_fork(config.module_init_seed):
                    trainer = get_trainer(trainer_name, config, env, algorithm)
                trainer.train(
                    previous_iterations=total_its,
                    validate_every=config.validate_every
                )

                total_its += trainer.iterations

                if not (i == len(trainer_names) - 1):
                    algorithm.save(f"post {trainer.name} save")

        # a final model save to the current checkpoint directory, even if no trainers were used
        algorithm.save("final save")

    algorithm.evaluate()

# ==============================================================================

if __name__ == "__main__":
    pre_hydra()
    main()
