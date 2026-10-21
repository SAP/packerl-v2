from flatdict import FlatDict
import itertools
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from wandb.wandb_run import Run as WandBRun
from PIL import Image

from ._wandb import _run_to_record, _atomic_write_json
from src.utils.constants import (ROLLOUT_LOG_PER_EP_FILENAME,
                                 ROLLOUT_LOG_ALL_IT_FILENAME,
                                 TRAINING_LOG_ALL_IT_FILENAME,
                                 VALIDATION_LOG_ALL_IT_FILENAME,
                                 VALIDATION_LOG_PER_EP_FILENAME,
                                 EVAL_LOG_ALL_EP_FILENAME,
                                 EVAL_LOG_PER_EP_FILENAME,
                                 DEFAULT_OUT_DIR)
from src.utils.features import ALL_GLOBAL_FEATURES
from src.utils.visualization import visualize_ep
from src.utils.utils import list_of_dict_to_dict_of_lists, dict_of_lists_to_dict_of_stats, to_flat_list


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB_START_METHOD",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def start_wandb_run(config: DictConfig) -> WandBRun:
    """
    Create a wandb logger with the given config and algorithm.
    Args:
        logger_config:

    Returns: A wandb logger to use.
    """
    reset_wandb_env()

    wandb_config = config.logging.wandb
    project_name = wandb_config.get("project_name")
    group_name = wandb_config.get("group_name")[-127:]
    job_type = wandb_config.get("job_type")[-63:]
    run_name = job_type[-63:] + "_" + wandb_config.get("run_name")[-63:]
    tags = wandb_config.get("tags", [])
    mode = "offline" if wandb_config.get("offline", False) else "online"
    if tags is None:
        tags = []
    if "idx" in config:
        run_name = (run_name + "_i" + str(config.idx))[-127:]
        job_type = (job_type + "_i" + str(config.idx))[-63:]
        tags.append("i" + str(config.idx))
    if "_version" in config:
        run_name = (run_name + "_v" + str(config._version))[-127:]
        job_type = (job_type + "_v" + str(config._version))[-63:]
        tags.append("v" + str(config._version))

    return wandb.init(config=OmegaConf.to_container(config, resolve=True),
                      project=project_name,  # Name of your WandB project
                      name=run_name,  # Name of the current run
                      group=group_name,  # Group name for the run
                      tags=tags,  # List of tags for your run
                      entity=wandb_config.get("entity"),  # WandB username or team name
                      settings=wandb.Settings(quiet=True),  # WandB settings
                      job_type=job_type,  # Name of your experiment
                      mode=mode,
                      dir=DEFAULT_OUT_DIR
                      )


# Global experiment recorder
_experiment_recorder = None


class ExperimentRecorder:
    """
    A helper class that handles the logging of training and evaluation outcomes to Weights and Biases and to disk.

    It is designed as a scoped context manager, initializing a W&B run on enter and finalizing it on exit.
    """
    def __init__(self, config: DictConfig):
        """
        Initializes the ExperimentRecorder with the given configuration, starting a Weights and Biases run if desired.
        """
        self.config = config
        self.log_path = Path(self.config.run_dir)

        # initialize W&B run if desired
        if self.config.logging.wandb.enabled:
            self._wandb_run = start_wandb_run(self.config)
        else:
            self._wandb_run = None

        self.rollout_log_all_it_fp = str((self.log_path / ROLLOUT_LOG_ALL_IT_FILENAME).resolve())
        self.training_log_all_it_fp = str((self.log_path / TRAINING_LOG_ALL_IT_FILENAME).resolve())
        self.validation_log_all_it_fp = str((self.log_path / VALIDATION_LOG_ALL_IT_FILENAME).resolve())
        self.eval_log_all_ep_fp = str((self.log_path / EVAL_LOG_ALL_EP_FILENAME).resolve())
        self.eval_log_per_ep_fp = str((self.log_path / EVAL_LOG_PER_EP_FILENAME).resolve())

    def finalize(self):
        """
        Finalizes the W&B run.
        """
        if self._wandb_run is not None:
            run_record = _run_to_record(self._wandb_run, is_sdk_run=True)
            self._wandb_run.finish(exit_code=0)
            record_path = Path(self._wandb_run.dir) / f"{self._wandb_run.id}.json"
            _atomic_write_json(record_path, run_record)

    def _prepare_episodes_for_logging(self, ep_summaries):
        """
        Prepares episodes for recording.
        """
        metrics_per_episode = []
        scenario_stats_per_episode = []
        times_per_episode = []
        vis_material_per_episode = []

        ep_metric_strs = ["global_reward", "local_reward", "oscillation_ratio_global",
                          "next_hop_spread_global", "il_matching_ratio",
                          "global_reward_normalized", "local_reward_normalized", "blended_reward"]
        ep_time_strs = ["inference_time_ms", "post_step_time_ms", "env_step_time_ms", "il_loss_calc_time"]

        for ep_summary in ep_summaries:

            # episode metrics (also sets up ep. metrics and highlights from provided (global) metrics)
            ep_metrics = {k: to_flat_list(ep_summary[k]) for k in ep_metric_strs if k in ep_summary}
            if "ep_metrics" in ep_summary:
                for metric_name, metric in ep_summary['ep_metrics'].items():
                    _, metric_type = ALL_GLOBAL_FEATURES[metric_name]
                    if metric_type == "b":  # bytes
                        metric /= 1e6
                        metric_name += " (MB)"
                    elif metric_type == "c":  # channel
                        metric /= 1e6
                        metric_name += " (MB, gross)"
                    elif metric_type == "d":  # latency
                        metric *= 1e3
                        metric_name += " (ms)"
                    elif metric_type in "trainable":  # ratio or load
                        metric *= 100
                        metric_name += " (%)"
                    else:  # ignore other global metrics like 'a' and 'u' types (shouldn't be here anyway, but just in case)
                        continue
                    ep_metrics[metric_name] = metric
                metrics_per_episode.append(ep_metrics)

            # scenario stats
            if "ep_scenario_stats" in ep_summary:
                scenario_stats_per_episode.append(dict(FlatDict(ep_summary['ep_scenario_stats'], delimiter='/')))

            # episode times
            ep_times = {k: to_flat_list(ep_summary[k]) for k in ep_time_strs if k in ep_summary}
            times_per_episode.append(ep_times)

            # visualization material
            vis_material_per_episode.append(ep_summary.get("ep_vis_material", {}))

        return metrics_per_episode, scenario_stats_per_episode, times_per_episode, vis_material_per_episode

    def _record_eps_metrics(self, metrics, scenario_stats, times, mode, it, log_per_ep_fp, log_all_it_fp):
        """
        Log metrics of multiple episodes to disk, and W&B if desired
        """

        # episode metrics as stats dict
        metrics = list_of_dict_to_dict_of_lists(metrics)
        if "global_reward" in metrics:
            metrics["global_reward"] = to_flat_list(metrics["global_reward"])
        if "local_reward" in metrics:
            metrics["local_reward"] = to_flat_list(metrics["local_reward"])
        if "il_matching_ratio" in metrics:
            metrics["il_matching_ratio"] = to_flat_list(metrics["il_matching_ratio"])
        if "global_reward_normalized" in metrics:
            metrics["global_reward_normalized"] = to_flat_list(metrics["global_reward_normalized"])
        if "local_reward_normalized" in metrics:
            metrics["local_reward_normalized"] = to_flat_list(metrics["local_reward_normalized"])
        if "blended_reward" in metrics:
            metrics["blended_reward"] = to_flat_list(metrics["blended_reward"])
        metric_stats = dict_of_lists_to_dict_of_stats(metrics)

        # scenario stats as lists (we don't log these to disk, only to W&B as histograms)
        scenario_stats = list_of_dict_to_dict_of_lists(scenario_stats)

        # episode times as stats dict
        times = list_of_dict_to_dict_of_lists(times)
        times["inference_time_ms"] = to_flat_list(times["inference_time_ms"])
        if "post_step_time_ms" in times:
            times["post_step_time_ms"] = to_flat_list(times["post_step_time_ms"])
        if "env_step_time_ms" in times:
            times["env_step_time_ms"] = to_flat_list(times["env_step_time_ms"])
        if "il_loss_calc_time" in times:
            times["il_loss_calc_time"] = to_flat_list(times["il_loss_calc_time"])
        time_stats = dict_of_lists_to_dict_of_stats(times)

        # log metrics and times to disk
        # - log metric/times lists (values per episode) for current iteration to a new file
        metrics_and_times = metrics | times
        mat_keys = list(metrics_and_times.keys())
        with open(log_per_ep_fp, "w") as f:
            f.write(f"{','.join(mat_keys)}\n")
            # get number of episodes from the metrics with the least number of entries
            num_episodes = min(len(metrics_and_times[k]) for k in mat_keys)
            for i in range(num_episodes):
                f.write(f"{','.join([str(metrics_and_times[k][i]) for k in mat_keys])}\n")

        # - log metric/times stats per iteration (mean, median, min, max, std): append to overall log file if exists
        metric_and_time_stats = metric_stats | time_stats
        mat_stat_keys = list(metric_and_time_stats.keys())
        if not os.path.isfile(log_all_it_fp):
            with open(log_all_it_fp, "w") as f:
                f.write(f"{','.join(mat_stat_keys)}\n")
        with open(log_all_it_fp, "a") as f:
            mats_comma_separated = ','.join([str(metric_and_time_stats[k]) for k in mat_stat_keys])
            f.write(f"{mats_comma_separated}\n")

        # log to W&B if desired
        if self._wandb_run is not None:
            wandb_logs = {}
            for metric_name, metric_value in metric_stats.items():
                wandb_logs[f"{mode}/metrics/{metric_name}"] = metric_value
            for scenario_stat_name, scenario_stat_values in scenario_stats.items():
                wandb_logs[f"{mode}/scenario_stats/{scenario_stat_name}"] = wandb.Histogram(scenario_stat_values)
            for time_name, time_value in time_stats.items():
                wandb_logs[f"{mode}/times/{time_name}"] = time_value

            if it is None:
                # if it is None, we are logging evaluation episodes, so we don't have an iteration number
                self._wandb_run.log(wandb_logs, commit=False)
            else:
                self._wandb_run.log(wandb_logs, commit=False, step=it)

    def _record_ep_vis(self, ep_vis, vis_dir, step_log_prefix, it):
        # log visualizations to disk and wandb if desired
        wandb_vis_dict = {}
        for vis_name, vis_data in ep_vis.items():

            vis_name_short = vis_name.split('_', 1)[1] if '_' in vis_name else vis_name

            # single PIL Image -> save as jpg and log as Wandb.Image
            if isinstance(vis_data, Image.Image):
                vis_path = (Path(vis_dir) / vis_name_short)
                vis_path_parent, vis_filename = vis_path.parent, f"{str(vis_path.stem)}.jpg"
                vis_path_parent.mkdir(parents=True, exist_ok=True)
                vis_data.save(vis_path_parent / vis_filename, optimize=True, quality=85)

                if self._wandb_run is not None:
                    wandb_vis_dict[f"{step_log_prefix}/vis/{vis_name_short}"] = wandb.Image(vis_data)

            # lists of PIL images -> create gif and log as Wandb.Image
            elif isinstance(vis_data, list):
                vis_path_parent = (Path(vis_dir) / vis_name_short)
                vis_type_name = vis_path_parent.parent.stem
                vis_path_parent.mkdir(parents=True, exist_ok=True)
                for i, vis in enumerate(vis_data):
                    vis_fp = str((vis_path_parent / f"{vis_type_name}{i:03d}.jpg").resolve())
                    vis.save(vis_fp, optimize=True, quality=85)
                vis_anim_fp = str((vis_path_parent / f"{vis_type_name}_anim.gif").resolve())
                vis_data[0].save(vis_anim_fp, save_all=True, append_images=vis_data[1:],
                                 duration=500, loop=0)

                if self._wandb_run is not None:
                    wandb_vis_dict[f"{step_log_prefix}/vis/{vis_name_short}"] = wandb.Video(vis_anim_fp, format="gif")

            # not sure what to do with this data type
            else:
                raise TypeError(f"Unsupported visualization data type: {type(vis_data)} for {vis_name}")

        # since metrics will always be logged thereafter, this does not commit
        if self._wandb_run is not None:
            if it is None:
                # if it is None, we are logging evaluation episodes, so we don't have an iteration number
                self._wandb_run.log(wandb_vis_dict, commit=False)
            else:
                self._wandb_run.log(wandb_vis_dict, commit=False, step=it)

    def _visualize_eps(self, episodes_vis_material, step_log_prefix, it):

        if "rollout" in step_log_prefix:
            return  # no visualization for rollout episodes

        # filter out empty vis dicts
        vis_eps = [ep_vis_material for ep_vis_material in episodes_vis_material if ep_vis_material]
        num_vis_ep = len(vis_eps)
        if num_vis_ep == 0:
            return  # no visualization

        if "validation" in step_log_prefix:
            vis_dir = str((self.log_path / f"it{it:03d}" / f"{step_log_prefix}_vis").resolve())
            ep_names = [f"{step_log_prefix}_it{it:03d}_ep{i:02d}" for i in range(num_vis_ep)]
        elif step_log_prefix == "evaluation":
            vis_dir = str((self.log_path / f"{step_log_prefix}_vis").resolve())
            ep_names = [f"{step_log_prefix}_ep{i:02d}" for i in range(num_vis_ep)]
        else:
            raise ValueError(f"Unknown step_log_prefix: {step_log_prefix}")

        Path(vis_dir).mkdir(parents=True, exist_ok=True)

        for vis_ep, ep_name in zip(vis_eps, ep_names):
            self._record_ep_vis(visualize_ep(vis_ep, ep_name), vis_dir, step_log_prefix, it)

    def _record_episodes(self, episodes_log, step_log_prefix, it, log_per_ep_fp, log_all_it_fp):
        """
        Log episodes to disk and W&B if desired
        """
        # Prepare the episodes for logging (e.g. adjusting metric names and values for readability)
        eps_metrics, eps_scenario_stats, eps_times, eps_vis_material = self._prepare_episodes_for_logging(episodes_log)

        # log episode metrics and highlights to disk, and W&B if desired
        self._record_eps_metrics(eps_metrics, eps_scenario_stats, eps_times,
                                 step_log_prefix, it, log_per_ep_fp, log_all_it_fp)

        # log visualizations of episodes to disk, and W&B if desired.
        # Number of episodes to visualize depends on config.
        self._visualize_eps(eps_vis_material, step_log_prefix, it)

    def _record_training_step(self, training_scalars, step_log_prefix, it):
        """
        Logs training scalars to disk and W&B if desired.
        """
        # log training scalar stats (mean, median, min, max, std) to disk. Append to overall log file if exists
        training_scalar_keys = list(training_scalars.keys())
        if not os.path.isfile(self.training_log_all_it_fp):
            with open(self.training_log_all_it_fp, "w") as f:
                f.write(f"{','.join(training_scalar_keys)}\n")
        with open(self.training_log_all_it_fp, "a") as f:
            scalars_comma_separated = ','.join([str(training_scalars[k]) for k in training_scalar_keys])
            f.write(f"{scalars_comma_separated}\n")

        if self._wandb_run is not None:
            self._wandb_run.log({f"{step_log_prefix}/{k}": v for k, v in training_scalars.items()},
                                commit=False, step=it)

    def record_trainer_iteration(self, it, iteration_log, trainer_name: str):
        """
        Logs the entire content of the given trainer iteration log to disk (and W&B if desired).
        """
        it_path = self.log_path / f"it{it:03d}_{trainer_name}"
        it_path.mkdir(parents=True, exist_ok=True)

        iteration_commit_log = dict()

        # log rollout to disk and W&B if desired
        rollout_log_per_ep_fp = str((it_path / ROLLOUT_LOG_PER_EP_FILENAME).resolve())
        iteration_commit_log["times/rollout_time_s"] = iteration_log.pop("rollout_time_s")
        self._record_episodes(episodes_log=iteration_log["rollout"],
                              step_log_prefix="rollout",
                              it=it,
                              log_per_ep_fp=rollout_log_per_ep_fp,
                              log_all_it_fp=self.rollout_log_all_it_fp,
                              )

        # if available, log validation to disk and W&B if desired
        if "validation" in iteration_log:
            validation_log_per_ep_fp = str((it_path / VALIDATION_LOG_PER_EP_FILENAME).resolve())
            iteration_commit_log["times/vali_time_s"] = iteration_log.pop("validation_time_s")
            self._record_episodes(episodes_log=iteration_log["validation"],
                                  step_log_prefix="validation",
                                  it=it,
                                  log_per_ep_fp=validation_log_per_ep_fp,
                                  log_all_it_fp=self.validation_log_all_it_fp,
                                  )

        # log training step to disk and W&B if desired.
        iteration_commit_log["times/train_time_s"] = iteration_log.pop("training_time_s")
        self._record_training_step(iteration_log["training_step"],
                                   step_log_prefix="training",
                                   it=it
                                   )

        # Finalizes W&B step (iteration) log
        if self._wandb_run is not None:
            self._wandb_run.log(iteration_commit_log, commit=True, step=it)

    def record_eval(self, eval_logs, eval_time_s):
        """
        Logs final evaluation results to disk (and W&B if desired).
        """
        eval_commit_log = {"times/eval_time_s": eval_time_s}

        self._record_episodes(episodes_log=eval_logs,
                              step_log_prefix="evaluation",
                              it=None,
                              log_per_ep_fp=self.eval_log_per_ep_fp,
                              log_all_it_fp=self.eval_log_all_ep_fp
                              )

        # runs are marked as finished in W&B whenever the recorder is finalized, which may also be due to
        # an error in the training/evaluation process. We log an additional marker to indicate that the run has
        # finished with presentable results.
        if self._wandb_run is not None:
            self._wandb_run.log(eval_commit_log, commit=True)


def setup_experiment_recorder(config: DictConfig):
    """
    Set up the experiment recorder.
    This function is called at the beginning of the training process to initialize the global experiment recorder.
    """
    global _experiment_recorder
    _experiment_recorder = ExperimentRecorder(config)

def RECORD_TRAINER_ITERATION(it, iteration_log, trainer_name: str):
    global _experiment_recorder
    if _experiment_recorder is None:
        raise RuntimeError("Experiment recorder has not been set up. Call setup_experiment_recorder first.")
    _experiment_recorder.record_trainer_iteration(it, iteration_log, trainer_name)

def RECORD_EVAL(eval_logs, eval_time_s):
    """
    Log final evaluation results to disk (and W&B if desired).
    """
    global _experiment_recorder
    if _experiment_recorder is None:
        raise RuntimeError("Experiment recorder has not been set up. Call setup_experiment_recorder first.")
    _experiment_recorder.record_eval(eval_logs, eval_time_s)

def finalize_experiment_recorder():
    """
    Finalize the experiment recorder.
    This function is called at the end of the training process to clean up resources.
    """
    global _experiment_recorder
    if _experiment_recorder is None:
        raise RuntimeError("Experiment recorder has not been set up. Call setup_experiment_recorder first.")
    _experiment_recorder.finalize()
