from typing import Any, Dict

from omegaconf import DictConfig
import torch
from torch import Tensor

from src.algorithm import get_algorithm
from src.algorithm.trainable import TrainableAlgorithm
from src.packerl_env import PackerlEnv
from src.scenario import NetworkScenario
from src.utils.logging import LOG_CONSOLE_INFO, LOG_CONSOLE_RUN
from src.utils.observations import Observation
from src.utils.torch import train_stat_dict
from src.utils.time import TimedBlock
from src.utils.types import ActionRaw, ActionSedi, Reward, LogprobTensor, ValueTensor
from src.trainer._trainer import Trainer


class ImitationLearningTrainer(Trainer):

    # only single-agent centralized training is supported for IL
    train_actor_loc = "single"
    train_actor_input = "birdseye"
    interact_with_env: bool  # whether to do IL during rollouts (True) or only BC on mock observations (False)
    learning_rate: float  # learning rate for the IL optimizer

    def __init__(self, config: DictConfig, trainer_config: DictConfig, env: PackerlEnv, algorithm: TrainableAlgorithm):
        super().__init__(config, trainer_config, env, algorithm)

        # Initialize the IL optimizer
        self.optimizer = torch.optim.Adam([{'params': self._algo.actor_params}], lr=self.learning_rate)

        # Initialize the expert algorithm
        LOG_CONSOLE_INFO("initializing expert algorithm")
        self.expert = get_algorithm(config, self._env,
                                    routing_algorithm=trainer_config.expert,
                                    load_dir=trainer_config.expert_load_dir,
                                    device=config.device
                                    )
        self.expert.set_loc(self.train_actor_loc, self.train_actor_input)
        self.expert_obs: Observation = None  # we will need to cache expert observations for the expert due to normalizer differences

        # whether to use interactive learning (IL during rollouts) or only behavioral cloning (BC on mock observations)
        if self.interact_with_env:
            self._name = "IL"
            self._run_il = self._run_interactive_il
        else:
            self._name = "BC"
            self._run_il = self._run_bc

    def _set_train(self, train: bool):
        """
        simple pass-through to the algorithm, because IL has no additional trainable components
        """
        self._algo.set_train(train)

    def _reset_env(self, scenario: NetworkScenario):
        # DON'T call algorithm.reset_env, because that already prepares and normalizes the obs,
        # while we'd like to pass the observation to the expert unmodified for it to process it in its own way.
        obs = self._env.reset(scenario)
        expert_obs = obs.clone()
        obs = self._algo._prepare_obs(obs)
        expert_obs = self.expert._prepare_obs(expert_obs)

        # if the expert is trainable, we need to reset its normalizer as well and do so using a clone of the env's
        # first observation, because the call to algorithm.obs_normalizer.reset(obs) will normalize the obs in-place.
        if isinstance(self.expert, TrainableAlgorithm):
            self.expert_obs = self.expert.obs_normalizer.normalize(expert_obs)
        else:
            self.expert_obs = expert_obs  # expert has no normalizer, just cache the obs
        obs = self._algo.obs_normalizer.reset(obs)
        return obs

    def _get_supervised_loss(self,
                             action_raw: Tensor,
                             action_sedi_sparse: Tensor,
                             expert_action_raw: Tensor,
                             expert_sedi_sparse: Tensor,
                             input_obs: Observation) -> Tensor:
        raise NotImplementedError

    def _compute_target_and_compare(self,
                                    obs: Observation,
                                    action_raw: ActionRaw,
                                    action_sedi: ActionSedi):

        # compute behavioral cloning target
        with torch.no_grad():
            expert_action_raw, expert_sedi, _, _, _ = self.expert.get_action(self.expert_obs)

        # Compute sparse representation of the (student/expert) sedi for loss computation and logging.
        # Since sedi are obtained per observation graph, but we're computing the loss over a batch of graphs,
        # we need to offset the sedi indices accordingly.
        obs_view = self._algo.get_actor_view(obs)
        NEs = torch.sum(obs_view.num_edges_per_graph * obs_view.num_nodes_per_graph).item()
        batching_offset = self._algo.get_action_sedi_offsets(obs_view)
        action_sedi_sparse = torch.zeros(NEs, device=action_sedi.device).scatter_(0, action_sedi + batching_offset, 1.0)
        expert_sedi_sparse = torch.zeros(NEs, device=expert_sedi.device).scatter_(0, expert_sedi + batching_offset, 1.0)

        with TimedBlock() as loss_timer:
            # compute the supervised loss for this step (depends on the specific algorithm implementation)
            step_loss = self._get_supervised_loss(action_raw, action_sedi_sparse,
                                                  expert_action_raw, expert_sedi_sparse,
                                                  obs_view)

        # sedi tensors can match for up to N**2 (= action_sedi.numel()) elements, so we compute the ratio
        matches = action_sedi_sparse * expert_sedi_sparse.detach().to(action_sedi_sparse.device)
        matching_ratio = matches.sum().item() / action_sedi.numel()

        return {
            "il_loss": step_loss,
            "il_loss_calc_time_ms": loss_timer.elapsed_ms,
            "il_matching_ratio": matching_ratio,
        }

    def post_il_step(self,
                     obs: Observation,
                     action_raw: ActionRaw,
                     action_sedi: ActionSedi,
                     logprob: LogprobTensor,
                     values: ValueTensor,
                     reward: Reward,
                     next_obs_raw: Observation,
                     done: bool,
                     infos: Dict[str, Any]) -> Observation:
        """
        Postprocessing step for the IL rollout phase. Calculates this step's
        behavioral cloning loss wrt./ the used SPF baseline prior to normalizing the next_obs.
        """
        infos |= self._compute_target_and_compare(obs, action_raw, action_sedi)

        # prepare and normalize next_obs, and cache the expert observation for the next step.
        # We have to clone next_obs for the expert before preparing/normalizing for the
        # student algorithm, because the normalizer will modify next_obs in-place.
        expert_next_obs_raw = next_obs_raw.clone()
        next_obs = self._algo._prepare_obs(next_obs_raw)
        expert_next_obs = self.expert._prepare_obs(expert_next_obs_raw)
        if isinstance(self.expert, TrainableAlgorithm):
            self.expert_obs = self.expert.obs_normalizer.normalize(expert_next_obs)  # don't update, just normalize
        else:
            self.expert_obs = expert_next_obs  # expert has no normalizer, just cache the next obs
        next_obs = self._algo.obs_normalizer.update_and_normalize(next_obs)
        return next_obs

    def _run_interactive_il(self):
        """
        Imitation Learning training iteration: collect full episodes using the environment
        """
        with TimedBlock() as rollout_timer:
            ep_summaries = self._algo.run_environment(env_mode="training",
                                                      scenarios=self._train_scenario_generator.generate(),
                                                      actor_loc=self.train_actor_loc,
                                                      actor_input=self.train_actor_input,
                                                      reset_func=self._reset_env,
                                                      get_action_func=self._algo.get_action,
                                                      post_step_func=self.post_il_step
                                                      )
        loss_list = []
        for ep_summary in ep_summaries:
            loss_list.extend(ep_summary.pop("il_loss"))  # pop loss from the episode summary
        il_loss = torch.mean(torch.stack(loss_list))

        return ep_summaries, round(rollout_timer.elapsed_s, 3), il_loss

    def _run_bc(self):
        """
        Behavioral Cloning training iteration: collect single timestep 'episodes' using mock observations
        """
        bc_step_infos = []
        bc_losses = []
        with TimedBlock() as rollout_timer:
            scenarios = self._train_scenario_generator.generate(with_events=False,
                                                                topology_multiplier=self.bc_topology_multiplier)
            self._algo.set_loc(self.train_actor_loc, self.train_actor_input)
            for scenario in scenarios:
                obs = self._reset_env(scenario)  # also prepares expert obs
                action_raw, action_sedi, _, _, inference_times_ms = self._algo.get_action(obs)
                bc_step_info = {
                    "inference_time_ms": inference_times_ms.mean().item(),
                } | self._compute_target_and_compare(obs, action_raw, action_sedi)
                bc_losses.append(bc_step_info.pop("il_loss"))
                bc_step_infos.append(bc_step_info)

        return bc_step_infos, round(rollout_timer.elapsed_s, 3), torch.mean(torch.stack(bc_losses))

    def _train_iteration(self, it: int):

        self._set_train(True)

        rollout_info, rollout_time_s, il_loss = self._run_il()

        # TRAINING STEP
        LOG_CONSOLE_RUN(f"--- training step ---")
        with TimedBlock() as train_timer:
            unclipped_gradients, clipped_gradients = self._opt_step(self.optimizer, il_loss)
        train_time_s = round(train_timer.elapsed_s, 3)

        train_step_summary = (
                {"il_loss": il_loss.item()}
                | train_stat_dict(unclipped_gradients, "unclipped_gradient")
                | train_stat_dict(clipped_gradients, "clipped_gradient")
        )

        self._set_train(False)

        return {
            "rollout": rollout_info,
            "training_step": train_step_summary,
            "rollout_time_s": rollout_time_s,
            "training_time_s": train_time_s,
        }