from omegaconf import DictConfig

from ._baseline import BaselineCentralizedAlgorithm
from .random_next_hop import RandomNextHopCentralizedAlgorithm
from .random_link_weight import RandomLinkWeightCentralizedAlgorithm
from .plf_oscillator import PLFOscillatorAlgorithm
from .spf import SPFCentralizedAlgorithm


def get_baseline_algorithm(config: DictConfig, env, routing_algorithm) -> BaselineCentralizedAlgorithm:
    """
    Returns:  An instance of a baseline routing algorithm based on the routing mode specified in the config.
    """
    if routing_algorithm == "random_next_hop":
        return RandomNextHopCentralizedAlgorithm(env)

    elif routing_algorithm == "random_link_weight":
        return RandomLinkWeightCentralizedAlgorithm(env)

    elif routing_algorithm == "spf":
        return SPFCentralizedAlgorithm(env, config.algorithm.spf.mode)

    elif routing_algorithm == "plf_oscillator":
        if config.eval_scenario.topology.graph.mode != "plf":
            raise ValueError("PLFOscillatorAlgorithm only supports 'plf' topology")
        return PLFOscillatorAlgorithm(env)

    elif routing_algorithm == "magnneto":
        magnneto_seed = int(config.algorithm.magnneto.seed)
        if not 1 <= magnneto_seed <= 16:
            raise ValueError("MagnnetoCentralizedAlgorithm only supports magnneto_seed values between 1 and 16")
        capacity_scaling = 0.001 * config.eval_scenario.ms_per_step  # MAGNNETO thinks in seconds
        # MAGNNETO uses tensorflow, which we don't want to import unless we really need it, so the algo import is here
        from src.algorithm.baselines.magnneto.magnneto import MagnnetoCentralizedAlgorithm
        return MagnnetoCentralizedAlgorithm(env, magnneto_seed, capacity_scaling)
    else:
        raise RuntimeError(f"Unknown routing algorithm '{routing_algorithm}' (not a baseline mode)")
