from __future__ import annotations
import torch
import torch.nn as nn


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "sin":
        # If you use sin activation, consider SIREN-style init (not included here).
        return torch.sin
    raise ValueError(f"Unsupported activation: {name}")


class TimeFourierEncoder(nn.Module):
    """
    Random Fourier features for scalar s in [0,1].
      phi(s) = [sin(2π s B), cos(2π s B)]
    """
    def __init__(self, out_dim: int = 64, sigma: float = 2.0):
        super().__init__()
        if out_dim % 2 != 0:
            raise ValueError("out_dim must be even")
        half = out_dim // 2
        B = torch.randn(1, half) * sigma
        self.register_buffer("B", B)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * torch.pi * (s @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FiLMGenerator(nn.Module):
    """
    Generate per-block FiLM params (alpha, beta) from conditioning (gamma, omega).
    """
    def __init__(self, hidden: int, num_blocks: int, cond_hidden: int, cond_depth: int, activation: str, film_scale: float):
        super().__init__()
        act = nn.SiLU() if activation.lower() in {"silu", "swish"} else nn.Tanh()
        layers = [nn.Linear(2, cond_hidden), act]
        for _ in range(cond_depth - 1):
            layers += [nn.Linear(cond_hidden, cond_hidden), act]
        layers += [nn.Linear(cond_hidden, num_blocks * 2 * hidden)]
        self.net = nn.Sequential(*layers)
        self.hidden = hidden
        self.num_blocks = num_blocks
        self.scale = float(film_scale)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, cond: torch.Tensor):
        out = self.net(cond).view(-1, self.num_blocks, 2, self.hidden)
        alpha = out[:, :, 0, :]
        beta  = out[:, :, 1, :]
        alpha = 1.0 + self.scale * alpha
        beta  = self.scale * beta
        return alpha, beta


class FiLMBlock(nn.Module):
    def __init__(self, hidden: int, activation: str, use_residual: bool):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden)
        self.act = nn.SiLU() if activation.lower() in {"silu", "swish"} else nn.Tanh()
        self.use_residual = bool(use_residual)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, h: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        z = self.act(self.fc(h))
        z = alpha * z + beta
        if self.use_residual:
            z = self.act(h + z)
        return z


class ToyOscillatorParamPINN(nn.Module):
    """
    Weak-ansatz parametric PINN for:
      x'' + 2γ x' + ω^2 x = 0

    We only enforce IC (x(t0)=x0, x'(t0)=v0) via a generic hard constraint:
      x(t) = x0 + v0*dt + gate(dt) * N_theta(t; γ, ω)

    where gate(dt) ~ O(dt^2) near dt=0, so IC holds exactly, but we DO NOT
    encode oscillatory structure (cos/sin).
    """
    def __init__(
        self,
        t0: float, t1: float,
        gamma_min: float, gamma_max: float,
        omega_min: float, omega_max: float,
        x0: float, v0: float,
        hidden_dim: int,
        num_blocks: int,
        activation: str,
        time_fourier_dim: int,
        time_fourier_sigma: float,
        use_residual: bool,
        cond_hidden: int,
        cond_depth: int,
        film_scale: float,
        gate_type: str = "s2",   # "s2" or "dt2"
        gate_tau: float = 1.0,   # used if gate_type == "exp2"
    ):
        super().__init__()
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.L  = float(t1 - t0)

        self.gamma_min, self.gamma_max = float(gamma_min), float(gamma_max)
        self.omega_min, self.omega_max = float(omega_min), float(omega_max)
        self.x0 = float(x0)
        self.v0 = float(v0)

        self.gate_type = gate_type
        self.gate_tau = float(gate_tau)

        self.enc = TimeFourierEncoder(out_dim=time_fourier_dim, sigma=time_fourier_sigma)

        # Include s explicitly (low-frequency), plus Fourier features (high-frequency)
        in_dim = 1 + time_fourier_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)

        self.blocks = nn.ModuleList([
            FiLMBlock(hidden_dim, activation=activation, use_residual=use_residual) for _ in range(num_blocks)
        ])

        self.out_proj = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.film = FiLMGenerator(hidden_dim, num_blocks, cond_hidden, cond_depth, activation, film_scale)
        self.act = nn.SiLU() if activation.lower() in {"silu", "swish"} else nn.Tanh()

    def _scale_cond(self, gamma: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        g = 2.0*(gamma - self.gamma_min)/(self.gamma_max - self.gamma_min) - 1.0
        w = 2.0*(omega - self.omega_min)/(self.omega_max - self.omega_min) - 1.0
        return torch.cat([g, w], dim=-1)

    def _gate(self, dt: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # Gate must satisfy gate(0)=0 and gate'(0)=0 to enforce x'(t0)=v0 exactly.
        if self.gate_type == "s2":
            return s**2
        if self.gate_type == "dt2":
            return dt**2
        if self.gate_type == "exp2":
            # (1-exp(-dt/tau))^2 ~ (dt/tau)^2 near 0, bounded at large dt.
            tau = torch.as_tensor(self.gate_tau, dtype=dt.dtype, device=dt.device)
            return (1.0 - torch.exp(-dt / tau))**2
        raise ValueError(f"Unknown gate_type: {self.gate_type}")

    def forward(self, t: torch.Tensor, gamma: torch.Tensor, omega: torch.Tensor):
        dt = t - self.t0
        s  = dt / self.L

        feats = torch.cat([s, self.enc(s)], dim=-1)  # [B, 1+d_ff]
        h = self.act(self.in_proj(feats))

        cond = self._scale_cond(gamma, omega)
        alpha, beta = self.film(cond)

        for i, blk in enumerate(self.blocks):
            h = blk(h, alpha[:, i, :], beta[:, i, :])

        n = self.out_proj(h)  # N_theta(t; gamma, omega)
        gate = self._gate(dt, s)

        x0 = torch.as_tensor(self.x0, dtype=t.dtype, device=t.device)
        v0 = torch.as_tensor(self.v0, dtype=t.dtype, device=t.device)

        x = x0 + v0 * dt + gate * n
        return x, n, gate, s
