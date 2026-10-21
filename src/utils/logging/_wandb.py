from __future__ import annotations

from datetime import datetime, date, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Dict, Optional, Sequence


def _as_plain_dict(x: Any) -> Any:
    """Best-effort turn mapping-like W&B objects into a plain dict."""
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, Mapping):
        return dict(x)
    # Some objects might have .items() but not register as Mapping
    try:
        items = getattr(x, "items", None)
    except Exception:
        items = None
    if callable(items):
        try:
            return dict(items())
        except Exception:
            pass
    return x


def _api_run_to_record(run: Any) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "id": getattr(run, "id", None),
        "name": getattr(run, "name", None),
        "state": getattr(run, "state", None),
        "entity": getattr(run, "entity", None),
        "project": getattr(run, "project", None),
        "path": getattr(run, "path", None),
        "url": getattr(run, "url", None),
        "created_at": getattr(run, "created_at", None),
        "notes": getattr(run, "notes", None),
        "tags": list(getattr(run, "tags", []) or []),
        "group": getattr(run, "group", None),
        "job_type": getattr(run, "job_type", None),
        "sweep_name": getattr(run, "sweep_name", None),
    }

    # Convert the “big ones” to plain dicts early.
    try:
        cfg = dict(getattr(run, "config", {}) or {})
        cfg = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
        rec["config"] = cfg
    except Exception as e:
        rec["config_error"] = repr(e)

    for attr in ("summary", "summary_metrics", "system_metrics", "metadata", "rawconfig"):
        try:
            rec[attr] = _as_plain_dict(getattr(run, attr))
        except Exception as e:
            rec[f"{attr}_error"] = repr(e)

    return rec


def _sdk_summary_to_plain(obj: Any) -> Any:
    """Convert wandb SDK Summary / SummarySubDict / dict-like objects to plain Python."""
    if obj is None:
        return None

    if isinstance(obj, Mapping):
        return {
            str(k): _sdk_summary_to_plain(v)
            for k, v in obj.items()
            if str(k) != "_wandb"
        }

    if isinstance(obj, (list, tuple)):
        return [_sdk_summary_to_plain(v) for v in obj]

    # Prefer the public dict-like surface first.
    if hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        try:
            return {
                str(k): _sdk_summary_to_plain(obj[k])
                for k in obj.keys()
                if str(k) != "_wandb"
            }
        except Exception:
            pass

    # Fallback for current wandb SDK Summary implementation.
    if hasattr(obj, "_as_dict") and callable(obj._as_dict):
        try:
            d = obj._as_dict()
            return {
                str(k): _sdk_summary_to_plain(v)
                for k, v in d.items()
                if str(k) != "_wandb"
            }
        except Exception:
            pass

    return obj


def _sdk_run_system_metrics(run: Any) -> Dict[str, Any]:
    # 1) If this SDK run (or a wrapper around it) exposes system metrics directly,
    # use that first.
    for attr in ("system_metrics", "_system_metrics"):
        value = getattr(run, attr, None)
        if value is not None:
            plain = _as_plain_dict(value)
            if plain:
                return plain

    # 2) Otherwise, best-effort fallback: pull the latest row from the system
    # history stream. This is the closest analogue to public-api system_metrics.
    history_fn = getattr(run, "history", None)
    if callable(history_fn):
        rows = history_fn(stream="system", pandas=False)

        # Some implementations may still return a dataframe-like object.
        if hasattr(rows, "to_dict"):
            try:
                rows = rows.to_dict(orient="records")
            except Exception:
                pass

        if isinstance(rows, Mapping):
            return _as_plain_dict(rows)

        if rows is not None:
            try:
                rows_list = list(rows)
            except TypeError:
                rows_list = []

            for row in reversed(rows_list):
                if row:
                    return _as_plain_dict(row)

    return {}


def _sdk_run_to_record(run: Any) -> Dict[str, Any]:
    created_at = getattr(run, "created_at", None)
    if created_at is None:
        start_time = getattr(run, "start_time", None)
        if start_time is not None:
            try:
                created_at = datetime.fromtimestamp(
                    float(start_time), tz=timezone.utc
                ).isoformat()
            except Exception:
                created_at = start_time

    rec: Dict[str, Any] = {
        "id": getattr(run, "id", None),
        "name": getattr(run, "name", None),
        "state": getattr(run, "state", None),  # often absent on SDK runs
        "entity": getattr(run, "entity", None),
        "project": getattr(run, "project", None),
        "path": getattr(run, "path", None),
        "url": getattr(run, "url", None),
        "created_at": created_at,
        "notes": getattr(run, "notes", None),
        "tags": list(getattr(run, "tags", ()) or ()),
        "group": getattr(run, "group", None),
        "job_type": getattr(run, "job_type", None),
        "sweep_name": getattr(run, "sweep_name", None),
    }

    try:
        cfg = dict(getattr(run, "config", {}) or {})
        cfg = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
        rec["config"] = cfg
    except Exception as e:
        rec["config_error"] = repr(e)

    # SDK run: summary is the main summary surface.
    try:
        summary = _sdk_summary_to_plain(getattr(run, "summary", None)) or {}
        rec["summary"] = summary
        rec["summary_metrics"] = summary
    except Exception as e:
        rec["summary_error"] = repr(e)
        rec["summary_metrics_error"] = repr(e)

    try:
        rec["system_metrics"] = _sdk_run_system_metrics(run)
    except Exception as e:
        rec["system_metrics_error"] = repr(e)

    # Optional: only include these if the SDK run actually exposes them.
    for attr in ("metadata", "rawconfig"):
        if hasattr(run, attr):
            try:
                rec[attr] = _as_plain_dict(getattr(run, attr))
            except Exception as e:
                rec[f"{attr}_error"] = repr(e)

    return rec


def _run_to_record(run: Any, is_sdk_run: bool = False) -> Dict[str, Any]:
    if is_sdk_run:
        return _sdk_run_to_record(run)
    else:
        return _api_run_to_record(run)


def _safe_to_json(x: Any) -> Optional[Any]:
    """Safely call x.to_json() if it exists and is callable. Never raises."""
    try:
        tj = getattr(x, "to_json", None)  # some W&B objects raise KeyError here -> catch broadly
    except Exception:
        return None
    if callable(tj):
        try:
            return tj()
        except Exception:
            return None
    return None


def _json_friendly(x: Any) -> Any:
    """Convert x into something json can handle. Must not throw."""
    try:
        if x is None or isinstance(x, (bool, int, float, str)):
            return x

        if isinstance(x, (datetime, date)):
            return x.isoformat()

        if isinstance(x, Path):
            return str(x)

        if isinstance(x, bytes):
            return {"__type__": "bytes", "length": len(x)}

        # Mapping-like (covers W&B Summary/config wrappers)
        if isinstance(x, Mapping):
            out: Dict[str, Any] = {}
            for k, v in x.items():
                out[str(k)] = _json_friendly(v)
            return out

        # Sequence-like (but not str/bytes which we already handled)
        if isinstance(x, Sequence):
            return [_json_friendly(v) for v in x]

        if isinstance(x, set):
            return [_json_friendly(v) for v in sorted(x, key=lambda z: str(z))]

        # Optional: numpy
        try:
            import numpy as np  # type: ignore

            if isinstance(x, np.generic):
                return x.item()
            if isinstance(x, np.ndarray):
                return x.tolist()
        except Exception:
            pass

        # W&B value types sometimes support to_json
        tj = _safe_to_json(x)
        if tj is not None:
            return _json_friendly(tj)

        # Last resort
        return repr(x)
    except Exception:
        # Absolutely never let serialization errors abort a run save
        return repr(x)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_json_friendly(data), f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)
