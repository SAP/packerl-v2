import os
from pathlib import Path
from typing import Iterable

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import torch


def _strip_override_prefix(s: str) -> str:
    # Hydra overrides can start with "+", "++", "~" etc.
    return s.lstrip("+~")


def sweeper_override_dirname(
    *,
    sanitize_path: bool = True,
    include_ablative_params: bool = True,
    _root_,
) -> str | None:
    """
    Return an override_dirname-like string, but containing ONLY the overrides whose
    keys appear in:
      - hydra.sweeper.grid_params
      - hydra.sweeper.list_params
    (optionally: hydra.sweeper.ablative_params)

    Then remove keys in hydra.job.config.override_dirname.exclude_keys.

    Returns None if nothing matched (so your shortener can fall back to "job_type").
    """
    root = _root_

    kv_sep = OmegaConf.select(root, "hydra.job.config.override_dirname.kv_sep", default="=")
    item_sep = OmegaConf.select(root, "hydra.job.config.override_dirname.item_sep", default=",")

    exclude_keys: Iterable[str] = OmegaConf.select(
        root, "hydra.job.config.override_dirname.exclude_keys", default=[]
    ) or []
    exclude = set(exclude_keys)

    # Keys defined as sweep axes in the sweeper config
    sweep_keys = set()
    for section in ("grid_params", "list_params"):
        params = OmegaConf.select(root, f"hydra.sweeper.{section}", default=None)
        if params:
            sweep_keys.update(params.keys())

    if include_ablative_params:
        abl = OmegaConf.select(root, "hydra.sweeper.ablative_params", default=[]) or []
        for d in abl:
            if isinstance(d, dict):
                sweep_keys.update(d.keys())
            else:
                # DictConfig behaves like a dict for .keys()
                sweep_keys.update(d.keys())

    # Per-job override list (same info as .hydra/overrides.yaml)
    overrides = OmegaConf.select(root, "hydra.overrides.task", default=[]) or []

    kept: list[str] = []
    for o in overrides:
        s = _strip_override_prefix(str(o).strip())
        if "=" not in s:
            continue
        key, val = s.split("=", 1)

        if key in sweep_keys and key not in exclude:
            if sanitize_path:
                key = key.replace("/", "_")
                val = val.replace("/", "_")
            kept.append(f"{key}{kv_sep}{val}")

    return item_sep.join(kept) if kept else None


def conditional_resolver(condition, if_true: str, if_false: str):
    if condition:
        return if_true
    else:
        return if_false


def shortener(input_string: str | None = None, length=3, show_config_stack=False):
    if input_string is None:
        return "job_type"
    output_parts = []

    for part in input_string.split(","):
        key, value = part.split("=")
        modified_key = ""

        key_parts = key.split(".")
        if not show_config_stack:
            key_parts = key_parts[-1:]
        for key_part in key_parts:
            for word in key_part.split("_"):
                modified_key += word[:length] + "_"
            modified_key = modified_key[:-1] + "."
        modified_key = modified_key[:-1]

        output_parts.append(f"{modified_key}={value}")

    return ",".join(output_parts)


def pre_hydra():
    """
    Called before hydra populates the config.
    """
    # full stack trace
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # load custom OmegaConf resolvers
    OmegaConf.register_new_resolver("sub_dir_shortener", shortener)
    OmegaConf.register_new_resolver("format", lambda inpt, formatter: formatter.format(inpt))
    OmegaConf.register_new_resolver("conditional_resolver", conditional_resolver)
    OmegaConf.register_new_resolver("sweeper_override_dirname", sweeper_override_dirname)


def update_config_with_derived_values(config: DictConfig):
    """
    Add derived values to the config, which are not set by the user but are needed for the algorithm to run.
    """

    # hydra configuration
    hydra_config = HydraConfig.get()
    # 1) path for saving the artifacts for specific runs
    config.run_dir = hydra_config.runtime.output_dir
    # 2) The underlying timestamped directory, used when loading existing models
    config.out_base_dir = Path(config.run_dir).parent.parent.stem
    # 3) Job IDs (for SLURM jobs)
    slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", None)
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    if slurm_array_job_id is not None and slurm_array_task_id is not None:
        config.job_id = f"{slurm_array_job_id}_{slurm_array_task_id}"
    elif slurm_job_id is not None:
        config.job_id = slurm_job_id
    elif hydra_config.job.get("id", None) is not None and hydra_config.job.get("num", None) is not None:
        config.job_id = f"{hydra_config.job.id}_{hydra_config.job.num}"
    else:
        config.job_id = "singleton"
    # 4) runtime choices from HydraConfig into actual config (remove leading underscore from keys)
    run_choices = hydra_config.runtime.choices
    run_choices_runs = {
        key.split("/", 1)[1][1:]: value
        for key, value in run_choices.items()
        if key.startswith("_runs/")
    }
    # add special key 'runs.full_scenario' as f"{runs.scenario}_{runs.scenario_mod}"
    if "scenario" in run_choices_runs and "scenario_mod" in run_choices_runs:
        run_choices_runs["full_scenario"] = f"{run_choices_runs['scenario']}_{run_choices_runs['scenario_mod']}"
    config.runs = OmegaConf.create(run_choices_runs)

    # mempool_keys for the shared memory provided by ns3-ai have to start from 1001
    config.mempool_key = os.getpid() + 1000

    # threading: if hydra.launcher.cpus_per_task (for MULTIRUNs) or just cpus_per_task (for local runs) is set,
    # we set pyTorch threads to that value minus 1 (at least 1) to leave one thread for the rest (incl. ns-3).
    cpus_per_task = hydra_config.launcher.get("cpus_per_task", config.get("cpus_per_task", 2))
    torch.set_num_threads(max(1, cpus_per_task - 1))
    config.num_torch_threads = torch.get_num_threads()

    # scenario config
    config.train_scenario.seed = config.train_scen_seed
    config.train_scenario.num_topologies = -1  # determined by specific trainer
    config.train_scenario.num_event_seqs_per_topology = -1  # determined by specific trainer
    config.train_scenario.num_event_seq_repeats = -1  # determined by specific trainer
    config.train_scenario.vis_n_topologies = 0  # we don't need visualization during training
    config.train_scenario.vis_n_event_seqs = 0  # we don't need visualization during training

    config.validation_scenario.seed = config.vali_scen_seed
    config.validation_scenario.num_topologies = config.validation_topologies
    config.validation_scenario.num_event_seqs_per_topology = config.validation_event_seqs_per_topology
    config.validation_scenario.num_event_seq_repeats = 1  # we don't repeat event sequences during validation
    config.validation_scenario.vis_n_topologies = config.visualization.vis_first_n_validation_topologies
    config.validation_scenario.vis_n_event_seqs = config.visualization.vis_first_n_validation_event_seqs

    config.eval_scenario.seed = config.eval_scen_seed
    config.eval_scenario.num_topologies = config.eval_topologies
    config.eval_scenario.num_event_seqs_per_topology = config.eval_event_seqs_per_topology
    config.eval_scenario.num_event_seq_repeats = 1  # we don't repeat event sequences during evaluation
    config.eval_scenario.vis_n_topologies = config.visualization.vis_first_n_evaluation_topologies
    config.eval_scenario.vis_n_event_seqs = config.visualization.vis_first_n_evaluation_event_seqs
