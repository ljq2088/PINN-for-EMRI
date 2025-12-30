from __future__ import annotations
from dataclasses import dataclass
import torch
from typing import Tuple


@dataclass
class UniformSampler:
    t0: float
    t1: float
    gamma_min: float
    gamma_max: float
    omega_min: float
    omega_max: float
    device: torch.device
    dtype: torch.dtype = torch.float32

    def sample_params(self, n: int):
        gamma = torch.rand((n, 1), device=self.device, dtype=self.dtype) * (self.gamma_max - self.gamma_min) + self.gamma_min
        omega = torch.rand((n, 1), device=self.device, dtype=self.dtype) * (self.omega_max - self.omega_min) + self.omega_min
        return gamma, omega

    def sample_t(self, n: int, t_low: float | None = None, t_high: float | None = None):
        lo = self.t0 if t_low is None else float(t_low)
        hi = self.t1 if t_high is None else float(t_high)
        t = torch.rand((n, 1), device=self.device, dtype=self.dtype) * (hi - lo) + lo
        return t

    def sample(self, n: int, t_low: float | None = None, t_high: float | None = None):
        t = self.sample_t(n, t_low=t_low, t_high=t_high)
        gamma, omega = self.sample_params(n)
        return t, gamma, omega



@dataclass
class ParamRange:
    gamma_min: float
    gamma_max: float
    omega_min: float
    omega_max: float


def _sample_params(n: int, pr: ParamRange, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.rand(n, 1, device=device) * (pr.gamma_max - pr.gamma_min) + pr.gamma_min
    w = torch.rand(n, 1, device=device) * (pr.omega_max - pr.omega_min) + pr.omega_min
    return g, w


def sample_uniform(n: int, t_max: float, pr: ParamRange, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.rand(n, 1, device=device) * float(t_max)
    g, w = _sample_params(n, pr, device)
    return t, g, w


def sample_exact_zero(n: int, pr: ParamRange, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.zeros(n, 1, device=device)
    g, w = _sample_params(n, pr, device)
    return t, g, w


def sample_near_zero(n: int, t_eps: float, pr: ParamRange, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.rand(n, 1, device=device) * float(t_eps)
    g, w = _sample_params(n, pr, device)
    return t, g, w
