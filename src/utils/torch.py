from contextlib import contextmanager
from typing import Dict, Any, List

import torch
from torch import Tensor
import torch_scatter


def scatter_logsumexp(values, idx):
    """
    Calculates the logsumexp of values over the given indices. Used e.g. to safely convert unnormalized logits
    into normalized probabilities.
    """
    max_per_index, _ = torch_scatter.scatter_max(values, idx)
    values_exp = torch.exp(values - max_per_index[idx])  # subtract max per index for numerical stability
    values_sum_exp = torch_scatter.scatter_add(values_exp, idx)
    values_logsumexp = torch.log(values_sum_exp) + max_per_index  # re-add max per index
    return values_logsumexp


def train_stat_dict(val: Tensor | Dict[Any, Tensor] | List, name):
    """
    Returns a dict of statistical scalars for the given tensor, with the given name as prefix.
    """
    if isinstance(val, dict):
        val_ = torch.cat(list(val.values()))
    elif isinstance(val, list):
        val_ = torch.tensor(val)
    elif val.numel() == 1:
        return {name: val.item()}
    else:
        val_ = val
    return {
        f"{name}_mean": val_.mean().item(),
        f"{name}_min": val_.min().item(),
        f"{name}_max": val_.max().item(),
        f"{name}_std": val_.std().item(),
    }

def common_values(values1: Tensor, values2: Tensor) -> Tensor:
    """
    Return a 1‐D tensor of all values that appear in both elems1 and elems2.
    Both inputs must be sorted ascending and of dtype torch.long.
    This version avoids out‐of‐bounds indexing.

    NOTE: this is faster than indices1[torch.isin(indices1, indices2)]
    """
    if values1.dtype != torch.long or values2.dtype != torch.long:
        raise TypeError("Both inputs must be of dtype torch.long")

    # 1) For each element in elems1, find where it would go in elems2:
    pos = torch.searchsorted(values2, values1)

    # 2) Make a boolean “valid” mask for positions strictly inside elems2:
    valid = pos < values2.numel()   # shape: (len(elems1),)

    # 3) Create an all‐False mask, then fill in only the “valid” slots:
    mask = torch.zeros_like(valid)
    if valid.any():
        # For those i where pos[i] < len(elems2),
        # check elems2[pos[i]] == elems1[i].
        eq = values2[pos[valid]] == values1[valid]
        # Place eq back into the correct positions of mask:
        mask[valid] = eq

    # 4) Select only the elems1[i] for which mask[i] is True:
    return values1[mask]


@contextmanager
def torch_seed_fork(seed: int | None, *, include_cuda: bool = True, devices=None):
    """
    If seed is not None, temporarily fork PyTorch RNG state (CPU + optionally CUDA) and (optionally) set a seed.
    Restores original RNG state on exit.

    Args:
        seed: If None -> no-op. If int -> fork RNG states and seed inside context.
        include_cuda: Whether to include CUDA RNG states (if CUDA is available).
        devices: Optional explicit CUDA device indices (e.g. [0,1]).
                 If None and include_cuda=True -> all visible CUDA devices.
    """
    if seed is None:
        yield
        return

    # Decide which CUDA devices to fork.
    if include_cuda and torch.cuda.is_available():
        if devices is None:
            devices = list(range(torch.cuda.device_count()))
    else:
        devices = []

    # fork_rng saves/restores RNG state for CPU and the specified CUDA devices.
    with torch.random.fork_rng(devices=devices, enabled=True):
        # Seed inside the fork (only affects the forked state).
        torch.manual_seed(seed)
        if devices:
            # Use manual_seed_all so any CUDA RNG usage is consistent across devices.
            torch.cuda.manual_seed_all(seed)
        yield
