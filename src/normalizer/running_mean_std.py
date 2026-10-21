from typing import Tuple
import torch
from torch import Tensor

from src.utils.canonical_bytes import canonical_bytes


class TorchRunningMeanStd:
    """
    Running mean and standard deviation computation for PyTorch tensors.
    """
    def __init__(self,
                 epsilon = 1e-10,
                 shape: Tuple[int, ...] = (),
                 dtype: torch.dtype = torch.double):
        """
        Creates a running mean and std object

        Args:
            epsilon: small value to avoid division by zero
            shape: shape of the data
            dtype: data type of the data
        """
        self.mean = torch.zeros(*shape, dtype=dtype)
        self.var = torch.ones(*shape, dtype=dtype)
        self.count = epsilon
        self.dtype = dtype

    def update(self, arr: Tensor) -> None:
        """
        Updates the running mean and std with a new batch of data

        Args:
            arr: new batch of data as a tensor
        """
        # keep only rows without any NaN
        arr = arr[~torch.isnan(arr).any(dim=1)]
        if arr.numel() == 0:
            return  # nothing to do if all rows had NaNs
        batch_mean = arr.mean(dim=0)
        if arr.shape[0] == 1:
            # For a single sample, set variance to zero.
            batch_var = torch.zeros_like(batch_mean)
        else:
            batch_var = arr.var(dim=0)
        batch_var = torch.nan_to_num(batch_var, nan=0.0)
        batch_count = arr.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: Tensor, batch_var: Tensor, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def to_canonical_hash(self) -> bytes:
        return canonical_bytes([self.mean, self.var, self.count, str(self.dtype)])
