from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class OscillatorIC:
    u0: float = 1.0
    v0: float = 0.0


def oscillator_residual(
    u: torch.Tensor,
    t: torch.Tensor,
    gamma: torch.Tensor,
    omega: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute residual r = u_tt + 2γ u_t + ω^2 u."""
    if t.ndim == 1:
        t = t[:, None]
    if gamma.ndim == 1:
        gamma = gamma[:, None]
    if omega.ndim == 1:
        omega = omega[:, None]
    if u.ndim == 1:
        u = u[:, None]

    if not t.requires_grad:
        t = t.requires_grad_(True)

    u_t = torch.autograd.grad(
        outputs=u,
        inputs=t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    u_tt = torch.autograd.grad(
        outputs=u_t,
        inputs=t,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    r = u_tt + 2.0 * gamma * u_t + (omega ** 2) * u
    return r, u_t, u_tt


def relative_residual(
    r: torch.Tensor,
    u: torch.Tensor,
    u_t: torch.Tensor,
    u_tt: torch.Tensor,
    gamma: torch.Tensor,
    omega: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Relative residual normalization."""
    denom = torch.abs(u_tt) + torch.abs(2.0 * gamma * u_t) + torch.abs((omega ** 2) * u) + eps
    return r / denom


def taylor_u_near0(
    t: torch.Tensor,
    gamma: torch.Tensor,
    omega: torch.Tensor,
    ic: OscillatorIC,
    order: int = 3,
) -> torch.Tensor:
    """ODE-implied Taylor expansion near t=0.

    u(0)=u0, u'(0)=v0
    u''(0) = -2γ v0 - ω^2 u0
    u'''(0) = -2γ u''(0) - ω^2 u'(0)
    """
    if t.ndim == 1:
        t = t[:, None]
    if gamma.ndim == 1:
        gamma = gamma[:, None]
    if omega.ndim == 1:
        omega = omega[:, None]

    u0 = torch.full_like(t, float(ic.u0))
    v0 = torch.full_like(t, float(ic.v0))

    a0 = -2.0 * gamma * v0 - (omega ** 2) * u0  # u''(0)
    out = u0 + v0 * t
    if order >= 2:
        out = out + 0.5 * a0 * t ** 2
    if order >= 3:
        j0 = -2.0 * gamma * a0 - (omega ** 2) * v0  # u'''(0)
        out = out + (1.0 / 6.0) * j0 * t ** 3
    return out


def analytic_solution(t: np.ndarray, gamma: float, omega: float, u0: float, v0: float) -> np.ndarray:
    """For plotting/testing only (NOT used in the model)."""
    t = np.asarray(t, dtype=float)
    if omega <= gamma:
        # over/critical damping
        d = math.sqrt(max(gamma * gamma - omega * omega, 0.0))
        r1 = -gamma + d
        r2 = -gamma - d
        if abs(r1 - r2) < 1e-12:
            # critical: u=(c1+c2 t) e^{-γ t}
            c1 = u0
            c2 = v0 + gamma * u0
            return (c1 + c2 * t) * np.exp(-gamma * t)
        else:
            # over: u = c1 e^{r1 t} + c2 e^{r2 t}
            c2 = (v0 - r1 * u0) / (r2 - r1)
            c1 = u0 - c2
            return c1 * np.exp(r1 * t) + c2 * np.exp(r2 * t)
    else:
        wd = math.sqrt(omega * omega - gamma * gamma)
        A = u0
        B = (v0 + gamma * u0) / wd
        return np.exp(-gamma * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))
