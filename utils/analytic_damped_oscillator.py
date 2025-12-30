from __future__ import annotations
import torch

def analytic_solution(t: torch.Tensor, gamma: torch.Tensor, omega: torch.Tensor, x0: float, v0: float, tol: float = 1e-10):
    t = t.reshape(*t.shape)
    gamma = torch.as_tensor(gamma, dtype=t.dtype, device=t.device)
    omega = torch.as_tensor(omega, dtype=t.dtype, device=t.device)
    x0_t = torch.tensor(x0, dtype=t.dtype, device=t.device)
    v0_t = torch.tensor(v0, dtype=t.dtype, device=t.device)

    delta = omega - gamma
    under = delta > tol
    over  = delta < -tol
    crit  = (~under) & (~over)

    x = torch.zeros_like(t)

    if under.any():
        Om = torch.sqrt(torch.clamp(omega**2 - gamma**2, min=0.0))
        exp = torch.exp(-gamma * t)
        A = x0_t
        B = (v0_t + gamma*x0_t) / (Om + 1e-30)
        x_u = exp*(A*torch.cos(Om*t) + B*torch.sin(Om*t))
        x = torch.where(under, x_u, x)

    if over.any():
        s = torch.sqrt(torch.clamp(gamma**2 - omega**2, min=0.0))
        r1 = -gamma + s
        r2 = -gamma - s
        C1 = (v0_t - r2*x0_t)/(r1-r2+1e-30)
        C2 = x0_t - C1
        x_o = C1*torch.exp(r1*t) + C2*torch.exp(r2*t)
        x = torch.where(over, x_o, x)

    if crit.any():
        x_c = torch.exp(-gamma*t)*(x0_t + (v0_t + gamma*x0_t)*t)
        x = torch.where(crit, x_c, x)

    return x
