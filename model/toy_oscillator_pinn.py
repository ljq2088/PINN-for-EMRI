from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, sigma: float = 2.0):
        super().__init__()
        if out_dim % 2 != 0:
            raise ValueError("fourier out_dim must be even")
        half = out_dim // 2
        B = torch.randn(in_dim, half) * sigma
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * torch.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, activation: nn.Module):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.act(x + h)

def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name in {"silu","swish"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")

@dataclass
class InputScaler:
    t0: float
    t1: float
    gamma_min: float
    gamma_max: float
    omega_min: float
    omega_max: float

    def __call__(self, t, gamma, omega):
        t_n = 2.0*(t-self.t0)/(self.t1-self.t0) - 1.0
        g_n = 2.0*(gamma-self.gamma_min)/(self.gamma_max-self.gamma_min) - 1.0
        w_n = 2.0*(omega-self.omega_min)/(self.omega_max-self.omega_min) - 1.0
        return torch.cat([t_n, g_n, w_n], dim=-1)

class ToyOscillatorPINN(nn.Module):
    def __init__(
        self,
        t0, t1,
        gamma_min, gamma_max,
        omega_min, omega_max,
        hidden_dim=64,
        num_layers=5,
        activation="tanh",
        use_fourier=True,
        fourier_dim=32,
        fourier_sigma=2.0,
        use_residual=True,
    ):
        super().__init__()
        self.scaler = InputScaler(t0,t1,gamma_min,gamma_max,omega_min,omega_max)
        act = _make_activation(activation)
        self.use_fourier = use_fourier

        in_dim = 3
        if use_fourier:
            self.ff = FourierFeatures(in_dim=in_dim, out_dim=fourier_dim, sigma=fourier_sigma)
            mlp_in = fourier_dim
        else:
            self.ff = None
            mlp_in = in_dim

        layers = [nn.Linear(mlp_in, hidden_dim), act]

        if use_residual:
            for _ in range(max(num_layers-2,1)):
                layers.append(ResidualMLPBlock(hidden_dim, act))
        else:
            for _ in range(max(num_layers-2,1)):
                layers.extend([nn.Linear(hidden_dim,hidden_dim), act])

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, gamma, omega):
        x = self.scaler(t,gamma,omega)
        if self.use_fourier:
            x = self.ff(x)
        return self.net(x)
