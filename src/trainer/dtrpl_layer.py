"""
Code for discrete Trust Region Projection Layers (TRPL).
Taken and amalgamated from https://github.com/pbecker93/discrete_trpl.
"""
from dataclasses import dataclass
from functools import partial
import math
from typing import Tuple, Callable, Union, Dict, Any, Optional

import torch
from torch import nn


EvalFunction = Callable[[torch.Tensor], torch.Tensor]
Interval = Tuple[torch.Tensor, torch.Tensor]  # (lower, upper)


class Optimizer1D:

    # Batched linspace: lower and upper are (batch_size, 1), returns (batch_size, num_points)
    @staticmethod
    def batched_linspace(lower: torch.Tensor, upper: torch.Tensor, num_points: int) -> torch.Tensor:
        # lower, upper: (batch_size, 1)
        steps = torch.linspace(0, 1, num_points, device=lower.device, dtype=lower.dtype).unsqueeze(0)  # (num_points,)
        # (batch_size, 1) + (batch_size, 1) * (num_points,) -> (batch_size, num_points)
        return lower + (upper - lower) * steps

    @staticmethod
    def _opt_step(func: EvalFunction, x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> Interval:
        batch_size, num_points = x.shape
        y = func(x)
        min_idx = torch.argmin(y, dim=1)
        l_idx = min_idx - 1
        u_idx = min_idx + 1
        l_tmp = x[torch.arange(batch_size), torch.clamp(l_idx, min=0, max=num_points - 1)]
        u_tmp = x[torch.arange(batch_size), torch.clamp(u_idx, min=0, max=num_points - 1)]
        new_lower = torch.where(l_idx < 0, lower.squeeze(1), l_tmp)
        new_upper = torch.where(u_idx >= num_points, upper.squeeze(1), u_tmp)
        return new_lower.unsqueeze(-1), new_upper.unsqueeze(-1)

    @staticmethod
    def optimize(func: EvalFunction,
                 lower: torch.Tensor,
                 upper: torch.Tensor,
                 num_points: int,
                 max_steps: int,
                 x_threshold: float) -> Tuple[torch.Tensor, dict]:

        l, u = lower, upper

        for step in range(max_steps):
            x = Optimizer1D.batched_linspace(l, u, num_points + 2)
            x = x[:, 1:-1]

            l, u = Optimizer1D._opt_step(func, x, l, u)

            if torch.all(torch.abs(l - u) < x_threshold):
                break

        x = (l + u) / 2

        return x, {"optim_steps": step + 1}


EMPTY_INFO_DICT = {
    # Empty dictionary containing dummy tensors for the expected keys. Used to match metric logging expectations.
    "violation": torch.tensor([0.0]),
    "final_kl": torch.tensor([0.0]),
    "initial_kl": torch.tensor([0.0]),
    "opt_eta": torch.tensor([0.0]),
    "projected_elements": 0,
    "projected_frac": 0.0,
    "optim_steps": 0,
}


@dataclass
class DtrplOptConfig:
    """Configuration for discrete trust region projection layer optimization.
        Example usages:
        DtrplOptConfig(num_points=32, max_steps=30, x_threshold=1e-5, lower=1e-5, upper=1e5)
        DtrplOptConfig.from_dict({"num_points": 32, "max_steps": 30, "x_threshold": 1e-5, "lower": 1e-5, "upper": 1e5})
        DtrplOptConfig(**cfg_dict) where cfg_dict is a dictionary with the appropriate keys.
        DtrplOptConfig.from_dict(cfg_dict)  # Returns default configuration
    """
    num_points: int = 32
    max_steps: int = 30
    x_threshold: float = 1e-6
    lower: float = 1e-6
    upper: float = 1e6

    @classmethod
    def from_dict(cls, opt_cfg: Optional[Union['DtrplOptConfig', Dict[str, Any]]]) -> 'DtrplOptConfig':
        """
        Create a DtrplOptConfig instance from various input types.

        Args:
            opt_cfg: None, DtrplOptConfig instance, or dict of parameters

        Returns:
            DtrplOptConfig instance
        """
        if opt_cfg is None:
            return cls()
        elif isinstance(opt_cfg, dict):
            return cls(**opt_cfg)
        elif isinstance(opt_cfg, cls):
            return opt_cfg
        else:
            raise TypeError(f"opt_cfg must be a {cls.__name__} instance, dict, or None. Got: {type(opt_cfg)}")


class _DtrplDualSolver(torch.autograd.Function):
    """
    Custom autograd function for discrete trust region projection
    """

    @staticmethod
    def _reps_dual(log_eta, bound, log_target_prob, log_ref_prob):
        eta = log_eta.exp()
        eta_ext = eta[..., None]
        inner = (eta_ext * log_ref_prob + log_target_prob) / (eta_ext + 1)
        log_integral = torch.logsumexp(inner, axis=-1)
        return bound * eta + (eta + 1) * log_integral

    @staticmethod
    def forward(ctx, log_target_prob, log_ref_prob, bound, opt_cfg=None):
        """
        Forward pass of the _DtrplDualSolver function.
        """
        opt_cfg = DtrplOptConfig.from_dict(opt_cfg)

        ctx.save_for_backward(log_target_prob, log_ref_prob)
        ctx.opt_cfg = opt_cfg

        lower = math.log(opt_cfg.lower) * torch.ones((log_target_prob.shape[0], 1), device=log_target_prob.device,
                                             dtype=log_target_prob.dtype)
        upper = math.log(opt_cfg.upper) * torch.ones((log_target_prob.shape[0], 1), device=log_target_prob.device,
                                             dtype=log_target_prob.dtype)

        dual = partial(
            _DtrplDualSolver._reps_dual,
            bound=bound,
            log_target_prob=log_target_prob.unsqueeze(1),
            log_ref_prob=log_ref_prob.unsqueeze(1)
        )

        # Find optimal eta using the Optimizer1D. We know that the optimal eta is in the range [lower, upper], and that
        # the dual function is convex in eta.
        opt_log_eta, stats_dict = Optimizer1D.optimize(
            dual,
            lower=lower,
            upper=upper,
            num_points=opt_cfg.num_points,
            max_steps=opt_cfg.max_steps,
            x_threshold=opt_cfg.x_threshold
        )
        opt_eta = opt_log_eta.exp()
        ctx.save_for_backward(log_target_prob, log_ref_prob, opt_eta)
        return opt_eta, stats_dict

    @staticmethod
    def backward(ctx, grad_output, _):
        log_target_prob, log_ref_prob, opt_eta = ctx.saved_tensors

        inner = (opt_eta * log_ref_prob + log_target_prob) / (opt_eta + 1)
        primal_solution = torch.softmax(inner, dim=-1)
        # d_primal_d_log_target_prob = torch.diag_embed(primal_solution) - torch.einsum('bi,bj->bij', primal_solution, primal_solution)
        # d_primal_d_eta_ref = torch.einsum('bij,bj->bi', d_primal_d_log_target_prob, log_ref_prob - log_target_prob) / ((opt_eta + 1)**2)
        diff = log_ref_prob - log_target_prob
        d_primal_d_eta = primal_solution * (diff - torch.einsum('bi,bi->b', primal_solution, diff)[..., None]) / (
                opt_eta + 1) ** 2

        # d_primal_d_log_target_prob = d_primal_d_log_target_prob / (opt_eta + 1)[..., None]
        # d_primal_d_eta = (d_primal_d_log_target_prob @ ((log_ref_prob - log_target_prob) / (opt_eta + 1)**2).unsqueeze(-1)).squeeze(-1) 

        v_temp = 1 + primal_solution.log() - log_ref_prob
        denom = torch.einsum('bi,bi->b', -v_temp, d_primal_d_eta)
        nom = primal_solution * (v_temp - (torch.einsum('bi,bi->b', primal_solution, v_temp)[..., None])) / (
                opt_eta + 1)

        # nom = torch.einsum('bij,bi->bj', d_primal_d_log_target_prob, v_temp)
        # nom = tmp / (opt_eta + 1) 
        # nom2 = torch.einsum('bij,bj->bi', d_primal_d_log_target_prob, v_temp)

        grad_log_target_prob = nom / denom.unsqueeze(-1) * grad_output
        grad_log_target_prob = torch.where(opt_eta > 10 * ctx.opt_cfg.lower, grad_log_target_prob,
                                           torch.zeros_like(grad_log_target_prob))

        # Return gradients for log_target_prob, log_ref_prob, and bound
        return grad_log_target_prob, None, None, None


class DtrplLayer(nn.Module):

    def __init__(self,
                 check_valid: bool = False,
                 opt_cfg: DtrplOptConfig = None):

        super(DtrplLayer, self).__init__()

        self.check_valid = check_valid
        self.compile = compile
        self.opt_cfg = opt_cfg if opt_cfg is not None else DtrplOptConfig()

    def forward(self, log_target_prob, log_ref_prob, bound: torch.Tensor | float, val_eps=1e-5):
        """
        Forward pass of the Dtrpl layer.
        """
        input_shape = log_target_prob.shape

        # flatten the input if it has more than 2 dimensions. We flatten to (*other_dims, num_logits)
        log_target_prob = log_target_prob.reshape(-1, input_shape[-1])
        log_ref_prob = log_ref_prob.reshape(-1, input_shape[-1])


        target_prob = log_target_prob.exp()

        if not isinstance(bound, torch.Tensor) or bound.numel() == 1:
            # broadcast float
            bound = bound * torch.ones((*target_prob.shape[:-1], 1), device=target_prob.device, dtype=target_prob.dtype)
        else:
            # flatten bound to match the shape of target_prob
            bound = bound.reshape(-1, 1)

        if self.check_valid:
            target_mass = torch.sum(target_prob, dim=-1)
            max_diff_target = torch.max(torch.abs(target_mass - 1))
            assert max_diff_target < val_eps, f"log_target_prob must be a valid distribution, given {max_diff_target=}"

            ref_mass = torch.sum(target_prob, dim=-1)
            max_diff_ref = torch.max(torch.abs(ref_mass - 1))
            assert max_diff_ref < val_eps, f"log_ref_prob must be a valid distribution, given {max_diff_ref=}"

        kl_div = (target_prob * (log_target_prob - log_ref_prob)).sum(dim=-1)
        needs_projection = kl_div >= bound[..., 0]
        if not torch.any(needs_projection):
            return log_target_prob.reshape(input_shape), EMPTY_INFO_DICT

        primal_solution = log_target_prob.clone()

        _log_target_prob = log_target_prob[needs_projection]
        _log_ref_prob = log_ref_prob[needs_projection]
        _bound = bound[needs_projection]

        _opt_eta, info_dict = _DtrplDualSolver.apply(_log_target_prob, _log_ref_prob, _bound, self.opt_cfg)
        inner = (_opt_eta * _log_ref_prob + _log_target_prob) / (_opt_eta + 1)
        _primal_solution = inner.log_softmax(dim=-1)

        primal_solution[needs_projection] = _primal_solution


        with torch.no_grad():
            final_kl_divergences = torch.sum(torch.exp(primal_solution) * (primal_solution - log_ref_prob), dim=-1)
            kl_violation = torch.clamp(final_kl_divergences - bound.squeeze(-1), min=0.0)
            info_dict["violation"] = kl_violation.reshape(input_shape[:-1]).detach()
            info_dict["final_kl"] = final_kl_divergences.reshape(input_shape[:-1]).detach()
            info_dict["initial_kl"] = kl_div.reshape(input_shape[:-1]).detach()

            opt_eta = torch.zeros((primal_solution.shape[0], 1),
                                  device=primal_solution.device,
                                  dtype=primal_solution.dtype)
            opt_eta[needs_projection] = _opt_eta
            info_dict["opt_eta"] = opt_eta.detach()
            info_dict["projected_elements"] = needs_projection.sum().item()
            info_dict["projected_frac"] = needs_projection.float().mean().item()


        # restore the original shape
        primal_solution = primal_solution.reshape(input_shape)
        return primal_solution, info_dict
