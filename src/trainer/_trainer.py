from copy import deepcopy

from omegaconf import DictConfig
from typing import List

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from src.algorithm.trainable import TrainableAlgorithm
from src.packerl_env import PackerlEnv
from src.scenario import Generator as ScenarioGenerator
from src.utils.logging import LOG_CONSOLE_RUN, RECORD_TRAINER_ITERATION


class Trainer:

    _name: str

    # from trainer config
    iterations: int
    num_topologies: int
    num_event_seqs_per_topology: int
    num_event_seq_repeats: int
    train_actor_loc: str
    train_actor_input: str
    max_grad_norm: float
    fixed_scenarios: bool

    # initialized by subclass
    optimizer: Optimizer

    def __init__(self,
                 config: DictConfig,
                 trainer_config: DictConfig,
                 env: PackerlEnv,
                 algorithm: TrainableAlgorithm):
        for k, v in trainer_config.items():
            setattr(self, k, v)
        self.num_episodes = (self.num_topologies * self.num_event_seqs_per_topology * self.num_event_seq_repeats)
        self._env = env
        self._algo = algorithm
        train_scenario_config = deepcopy(config.train_scenario)
        train_scenario_config.num_topologies = self.num_topologies
        train_scenario_config.num_event_seqs_per_topology = self.num_event_seqs_per_topology
        train_scenario_config.num_event_seq_repeats = self.num_event_seq_repeats
        self._train_scenario_generator = ScenarioGenerator(train_scenario_config)

        self.ep_length = config.train_scenario.ep_length

    @property
    def name(self):
        return self._name

    def __str__(self):
        return (f"{self._name}, {self.iterations=}, episodes={self.num_episodes} "
                f"({self.num_topologies=} topologies "
                f"x {self.num_event_seqs_per_topology=} event seq.s "
                f"x {self.num_event_seq_repeats=} seq. repeats)")

    def _set_train(self, train: bool):
        raise NotImplementedError  # implemented by subclass

    @classmethod
    def get_optimizer_param_list(cls, optimizer: Optimizer) -> List[Parameter]:
        return [p for g in optimizer.param_groups for p in g['params']]

    @classmethod
    def get_optimizer_flat_grads(cls, optimizer: Optimizer) -> Tensor:
        return torch.cat([p.grad.flatten() for p in cls.get_optimizer_param_list(optimizer) if p.grad is not None])

    def _opt_step(self, optimizer: Optimizer, loss: Tensor) -> (Tensor, Tensor):
        optimizer.zero_grad()
        loss.backward()
        unclipped_grads = self.get_optimizer_flat_grads(optimizer)
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.get_optimizer_param_list(optimizer), self.max_grad_norm)
        clipped_grads = self.get_optimizer_flat_grads(optimizer)
        optimizer.step()
        return unclipped_grads, clipped_grads

    def _train_iteration(self, it: int):
        raise NotImplementedError

    def train(self, previous_iterations: int, validate_every: int):
        """
        Train the algorithm for a number of iterations, each consisting of a number of episodes.
        After every 'validate_every' iterations (counted across all trainers), validate the algorithm.
        """
        for it in range(1, self.iterations + 1):
            LOG_CONSOLE_RUN("")
            LOG_CONSOLE_RUN(f"=== {self._name} iteration {it}/{self.iterations} ===")
            if self.fixed_scenarios:
                # keep the same scenarios for each iteration
                self._train_scenario_generator.reset_rng()
            iteration_log = self._train_iteration(it)
            total_it = it + previous_iterations
            if total_it % validate_every == 0:
                LOG_CONSOLE_RUN(f"--- validation after {self._name} iteration {it}/{self.iterations} ---")
                self._algo.validate(iteration_log)
            LOG_CONSOLE_RUN(f"--- saving records ---")
            RECORD_TRAINER_ITERATION(total_it, iteration_log, self._name)
