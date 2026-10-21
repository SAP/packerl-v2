from omegaconf import DictConfig

from .algorithm import Algorithm
from src.scenario import Generator as ScenarioGenerator
from .baselines import get_baseline_algorithm
from .trainable import create_or_load_trainable_algorithm
from src.utils.logging import LOG_CONSOLE_INFO


def get_algorithm(config: DictConfig, env, routing_algorithm: str = None, load_dir: str = None, device: str = "cpu") \
        -> Algorithm:
    """
    Get the algorithm based on the routing mode specified in the config.
    """
    algo_name = routing_algorithm or config.algorithm.routing_algorithm
    if algo_name == "trainable":
        l_dir = load_dir or config.algorithm.trainable.load_dir
        algo = create_or_load_trainable_algorithm(config, env, l_dir, device)
        LOG_CONSOLE_INFO(f"initialized trainable algorithm: {algo.name}")
    else:
        algo = get_baseline_algorithm(config, env, algo_name)
        LOG_CONSOLE_INFO(f"initialized baseline algorithm: {algo_name}")

    # set validation and evaluation parameters
    algo._vali_scenario_generator = ScenarioGenerator(config.validation_scenario)
    algo._eval_scenario_generator = ScenarioGenerator(config.eval_scenario)
    algo._eval_actor_loc = config.eval_agent_loc
    algo._eval_actor_input = config.eval_observer_loc
    return algo
