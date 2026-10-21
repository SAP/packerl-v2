from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import wandb

from src.utils.constants import WANDB_PROJECT_NAME


BASELINE_COLORS = {
    "rip": "#949494",  # gray
    "eigrp": "#ca9161",  # brown
    "ospf": "#fbafe4",  # pink
}

# dash patterns to cycle through when several baselines collide
BASELINE_DASHES = {
    "ospf": (4, 2),         # ---- ---- ---- ---- ----
    "rip": (2, 2),       # --  --  --  --  --  --
    "eigrp": (3, 1, 1, 1),  # --- - --- - --- - --- -
}

BASELINE_LEGEND_NAMES = {
    "rip": "\spr",
    "eigrp": "\spe",
    "ospf": "\spo",
}

SNS_CB_COLORS = [
    '#0173b2',  # blue [LOGGIA]
    '#de8f05',  # orange [M-Slim]
    '#029e73',  # green [LOGGIA no IL]
    '#d55e00',  # red orange [FieldLines]
    '#cc78bc',  # violet
    '#ca9161',  # brown [LOGGIA-path]
    '#fbafe4',  # pink [MAGNNETO]
    '#949494',  # gray [IL]
    '#ece133',  # yellow
    '#56b4e9',  # light blue [LOGGIA-PPO]
]


def _sanitize_folder_name(s: str) -> str:
    s = s.strip().replace("/", "__")
    s = re.sub(r"[^a-zA-Z0-9._\-+=@]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _parse_dt(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    if isinstance(x, datetime):
        return x
    if isinstance(x, str):
        # handle "2025-01-01T12:34:56" and "2025-01-01T12:34:56Z"
        s = x.strip()
        if s.endswith("Z"):
            s = s[:-1]
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None
    return None


def _get_by_dotted(obj: Any, dotted: str) -> Any:
    """
    Supports keys like:
      - "config.exp_name"
      - "summary.acc"
      - "created_at"
    """
    cur = obj
    for part in dotted.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            # allow attribute fallback (rare)
            cur = getattr(cur, part, None)
    return cur


def _match_scalar(val: Any, cond: Any) -> bool:
    # list membership convenience:
    if isinstance(val, list) and not isinstance(cond, dict):
        return cond in val
    return val == cond


def _cmp_prepare(val: Any) -> Any:
    # Try datetime compare for ISO strings
    dv = _parse_dt(val)
    return dv if dv is not None else val


def _match_ops(val: Any, ops: Dict[str, Any]) -> bool:
    """
    Minimal Mongo-like operator support:
      $in, $nin, $exists, $regex, $gte, $gt, $lte, $lt
    """
    if "$exists" in ops:
        exists = ops["$exists"]
        if bool(val is not None) != bool(exists):
            return False

    if "$regex" in ops:
        if val is None:
            return False
        pattern = ops["$regex"]
        flags = 0
        if ops.get("$options") and "i" in str(ops["$options"]):
            flags |= re.IGNORECASE
        if re.search(pattern, str(val), flags=flags) is None:
            return False

    if "$in" in ops:
        choices = ops["$in"]
        if isinstance(val, list):
            if not any(v in choices for v in val):
                return False
        else:
            if val not in choices:
                return False

    if "$nin" in ops:
        choices = ops["$nin"]
        if isinstance(val, list):
            if any(v in choices for v in val):
                return False
        else:
            if val in choices:
                return False

    # Comparisons
    v = _cmp_prepare(val)
    if "$gte" in ops:
        if v is None or v < _cmp_prepare(ops["$gte"]):
            return False
    if "$gt" in ops:
        if v is None or v <= _cmp_prepare(ops["$gt"]):
            return False
    if "$lte" in ops:
        if v is None or v > _cmp_prepare(ops["$lte"]):
            return False
    if "$lt" in ops:
        if v is None or v >= _cmp_prepare(ops["$lt"]):
            return False

    return True


def _match_filter_dict(run_dict: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    """
    Evaluates W&B-like filters against the serialized run dict.
    Supports $and / $or at top-level and nested comparisons via dotted keys.
    """
    if "$and" in flt:
        parts = flt["$and"]
        return all(_match_filter_dict(run_dict, p) for p in parts if isinstance(p, dict))
    if "$or" in flt:
        parts = flt["$or"]
        return any(_match_filter_dict(run_dict, p) for p in parts if isinstance(p, dict))

    for key, cond in flt.items():
        if key in ("$and", "$or"):
            continue

        val = _get_by_dotted(run_dict, key)

        if isinstance(cond, dict) and any(k.startswith("$") for k in cond.keys()):
            if not _match_ops(val, cond):
                return False
        else:
            if not _match_scalar(val, cond):
                return False

    return True


@dataclass(frozen=True)
class RunLike:
    """
    Lightweight, plotting-friendly stand-in for wandb.apis.public.Run.
    """
    _d: Dict[str, Any]

    @property
    def id(self) -> Optional[str]:
        return self._d.get("id")

    @property
    def name(self) -> Optional[str]:
        return self._d.get("name")

    @property
    def entity(self) -> Optional[str]:
        return self._d.get("entity")

    @property
    def project(self) -> Optional[str]:
        return self._d.get("project")

    @property
    def path(self) -> Any:
        return self._d.get("path")

    @property
    def url(self) -> Optional[str]:
        return self._d.get("url")

    @property
    def state(self) -> Optional[str]:
        return self._d.get("state")

    @property
    def created_at(self) -> Any:
        return self._d.get("created_at")

    @property
    def tags(self) -> List[str]:
        return list(self._d.get("tags") or [])

    @property
    def group(self) -> Any:
        return self._d.get("group")

    @property
    def job_type(self) -> Any:
        return self._d.get("job_type")

    @property
    def sweep_name(self) -> Any:
        return self._d.get("sweep_name")

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._d.get("config") or {})

    @property
    def summary(self) -> Dict[str, Any]:
        # Your crawler stores both sometimes; pick the most useful
        if isinstance(self._d.get("summary"), dict):
            return dict(self._d["summary"])
        if isinstance(self._d.get("summary_metrics"), dict):
            return dict(self._d["summary_metrics"])
        return {}

    @property
    def summary_metrics(self) -> Dict[str, Any]:
        return dict(self._d.get("summary_metrics") or {})

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._d.get("metadata") or {})

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._d)


class CachedApi:
    """
    Drop-in replacement for wandb.Api for the subset you need:
      - runs(path, filters=...)
    Uses your crawler's output folder structure.
    """

    def __init__(self, destination: Union[str, Path]):
        self.destination = Path(destination).expanduser().resolve()
        self._project_cache: Dict[str, List[RunLike]] = {}

    def _project_dir(self, path: str) -> Path:
        # path is "entity/project"
        return self.destination / _sanitize_folder_name(path)

    def _load_project(self, path: str) -> List[RunLike]:
        if path in self._project_cache:
            return self._project_cache[path]

        pdir = self._project_dir(path)
        if not pdir.exists():
            raise FileNotFoundError(f"Cached project folder not found: {pdir}")

        runs: List[RunLike] = []
        for fp in pdir.rglob("*.json"):
            if fp.name.startswith("_"):
                continue
            try:
                d = json.loads(fp.read_text(encoding="utf-8"))
                # ensure config/summary exist as dicts for dot-filtering
                if not isinstance(d.get("config"), dict):
                    d["config"] = {}
                if not isinstance(d.get("summary"), dict) and isinstance(d.get("summary_metrics"), dict):
                    d["summary"] = d["summary_metrics"]
                runs.append(RunLike(d))
            except Exception:
                # If you want, you can log bad JSON files here.
                continue

        self._project_cache[path] = runs
        return runs

    def runs(self, path: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, **_: Any) -> List[RunLike]:
        all_runs = self._load_project(path)
        if not filters:
            return list(all_runs)

        # Evaluate filters against the underlying dict
        out: List[RunLike] = []
        for r in all_runs:
            if _match_filter_dict(r.to_dict(), filters):
                out.append(r)
        return out


def crawl_results_wandb(api: wandb.Api, args: argparse.Namespace, exp_module):
    print("Crawling results from wandb...")
    projects = WANDB_PROJECT_NAME if not hasattr(exp_module, "project") else exp_module.project
    if not isinstance(projects, list):
        projects = [projects]

    run_filters = getattr(exp_module, "run_filters", {}) or {}

    all_runs = []
    for entity in exp_module.entities:
        for exp_name in exp_module.exp_names:
            for proj in projects:
                path = f"{entity}/{proj}"
                combined_filters = run_filters | {"config.exp_name": exp_name}

                print(f" - {entity=}, {proj=}, {exp_name=}, {run_filters=}")
                runs = api.runs(path=path, filters=combined_filters)

                if getattr(args, "debug", False):
                    print(f"Found {len(runs)} runs for this combination.")
                all_runs.extend(runs)
    return all_runs


def crawl_results_cached(api: CachedApi, args: argparse.Namespace, exp_module):
    print("Crawling results cache...")

    run_filters = getattr(exp_module, "run_filters", {}) or {}

    all_runs = []
    for exp_name in exp_module.exp_names:
        combined_filters = run_filters | {"config.exp_name": exp_name}

        print(f" - {args.cached_dir=}, {exp_name=}, {run_filters=}")
        runs = api.runs(filters=combined_filters)

        if getattr(args, "debug", False):
            print(f"Found {len(runs)} runs for this combination.")
        all_runs.extend(runs)

    return all_runs

def crawl_results(args: argparse.Namespace, exp_module):
    """
    Crawls results from either:
      - local cached runs (if args.cached_dir is provided), or
      - live W&B API (otherwise)

    Returns: a list of run-like objects (wandb.Run or RunLike from cache)
    """
    if getattr(args, "cached_dir", None):
        api = CachedApi(args.cached_dir)
        using_cached_api = True
    else:
        api = wandb.Api(timeout=600)
        using_cached_api = False

    projects = WANDB_PROJECT_NAME if not hasattr(exp_module, "project") else exp_module.project
    if not isinstance(projects, list):
        projects = [projects]

    run_filters = getattr(exp_module, "run_filters", {}) or {}

    all_runs = []
    for entity in exp_module.entities:
        for exp_name in exp_module.exp_names:
            for proj in projects:
                path = "" if using_cached_api else f"{entity}/{proj}"
                combined_filters = run_filters | {"config.exp_name": exp_name}

                print(f" - {entity=}, {proj=}, {exp_name=}, {run_filters=}")
                runs = api.runs(path=path, filters=combined_filters)

                if getattr(args, "debug", False):
                    print(f"Found {len(runs)} runs for this combination.")
                all_runs.extend(runs)

    return all_runs


def get_1metric_run_grid(all_runs, args: argparse.Namespace, exp_module):

    print("Preparing (filtered) run grid...")

    # args.matrix_config is the name of the config file in scripts/plot_config without the .py extension
    matrix_rows = exp_module.rows
    matrix_cols = exp_module.cols
    num_rows, num_cols = len(matrix_rows), len(matrix_cols)
    matrix_row_names = [row[0] for row in matrix_rows]
    matrix_col_names = [col[0] for col in matrix_cols]
    metric_name = exp_module.metric_name

    # prepare structure of runs
    all_run_tuples = []
    for r in all_runs:
        # filter out runs that did not finish
        if metric_name not in r.summary:
            if args.debug:
                print(f"Run {r.id} did not finish.")
            continue
        r_config = dict(sorted(r.config.items()))
        r_metric = r.summary[metric_name]
        all_run_tuples.append((r_config, r_metric))

    # create run matrix (list of lists, len(matrix_rows) rows and len(matrix_cols) columns) containing the runs
    run_matrix = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
    for i in range(num_rows):
        row_name, row_filter_func = matrix_rows[i]
        for j in range(num_cols):
            col_name, col_filter_func = matrix_cols[j]
            for run_config, run_metric in all_run_tuples:
                if row_filter_func(run_config) and col_filter_func(run_config):
                    run_matrix[i][j].append(run_metric)
            if args.debug:
                print(f"Cell ({i}, {j}) - {row_name} / {col_name}: Found {len(run_matrix[i][j])} runs.")

    return run_matrix, matrix_row_names, matrix_col_names, metric_name
