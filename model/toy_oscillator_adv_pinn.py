from __future__ import annotations
import torch
import torch.nn as nn

def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")

class TimeFourierEncoder(nn.Module):
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
    def __init__(self, hidden: int, num_blocks: int, cond_hidden: int, cond_depth: int, activation: str, film_scale: float):
        super().__init__()
        act = _act(activation)
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
        self.act = _act(activation)
        self.use_residual = bool(use_residual)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, h: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        z = self.act(self.fc(h))
        z = alpha * z + beta
        if self.use_residual:
            z = self.act(h + z)
        return z

class ToyOscillatorAdvPINN(nn.Module):
    """
    Phase-aware ansatz (underdamped):
    x(t)=exp(-γΔt)[(Cc + s^2 A)cos(ΩΔt) + (Cs + s^2 B)sin(ΩΔt)],
    Ω=sqrt(ω^2-γ^2), s=Δt/L.

    This enforces x(0)=x0 and x'(0)=v0 exactly.
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
    ):
        super().__init__()
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.L  = float(t1 - t0)

        self.gamma_min, self.gamma_max = float(gamma_min), float(gamma_max)
        self.omega_min, self.omega_max = float(omega_min), float(omega_max)
        self.x0 = float(x0)
        self.v0 = float(v0)

        self.enc = TimeFourierEncoder(out_dim=time_fourier_dim, sigma=time_fourier_sigma)
        self.in_proj = nn.Linear(time_fourier_dim, hidden_dim)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)

        self.blocks = nn.ModuleList([
            FiLMBlock(hidden_dim, activation=activation, use_residual=use_residual) for _ in range(num_blocks)
        ])

        # output two channels: A(s), B(s)
        self.out_proj = nn.Linear(hidden_dim, 2)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.film = FiLMGenerator(hidden_dim, num_blocks, cond_hidden, cond_depth, activation, film_scale)
        self.act = _act(activation)

    def _scale_cond(self, gamma: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        g = 2.0*(gamma - self.gamma_min)/(self.gamma_max - self.gamma_min) - 1.0
        w = 2.0*(omega - self.omega_min)/(self.omega_max - self.omega_min) - 1.0
        return torch.cat([g, w], dim=-1)

    def forward(self, t: torch.Tensor, gamma: torch.Tensor, omega: torch.Tensor):
        dt = t - self.t0
        s  = dt / self.L

        te = self.enc(s)
        h = self.act(self.in_proj(te))

        cond = self._scale_cond(gamma, omega)
        alpha, beta = self.film(cond)

        for i, blk in enumerate(self.blocks):
            h = blk(h, alpha[:, i, :], beta[:, i, :])

        AB = self.out_proj(h)
        A = AB[:, [0]]
        B = AB[:, [1]]

        # underdamped frequency
        Omega = torch.sqrt(torch.clamp(omega**2 - gamma**2, min=1e-30))

        x0 = torch.as_tensor(self.x0, dtype=t.dtype, device=t.device)
        v0 = torch.as_tensor(self.v0, dtype=t.dtype, device=t.device)

        Cc = x0
        Cs = (v0 + gamma*x0) / Omega

        corr = s**2
        cos = torch.cos(Omega * dt)
        sin = torch.sin(Omega * dt)

        x = torch.exp(-gamma * dt) * ((Cc + corr*A) * cos + (Cs + corr*B) * sin)
        return x, A, B, s, Omega
