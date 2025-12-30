from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioned on a parameter vector.

    Given z (N, z_dim), outputs per-layer (gamma, beta) each of shape (N, hidden_dim).
    Then hidden activation h is modulated as:
        h <- (1 + gamma) * h + beta
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        n_layers: int,
        width: int = 128,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)

        if act == "silu":
            activation = nn.SiLU()
        elif act == "tanh":
            activation = nn.Tanh()
        else:
            raise ValueError(act)

        self.mlp = nn.Sequential(
            nn.Linear(self.z_dim, width),
            activation,
            nn.Linear(width, width),
            activation,
            nn.Linear(width, self.n_layers * 2 * self.hidden_dim),
        )

    def forward(self, z: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if z.ndim != 2 or z.shape[-1] != self.z_dim:
            raise ValueError(f"Expected z with shape (N,{self.z_dim}), got {tuple(z.shape)}")
        out = self.mlp(z)
        out = out.view(z.shape[0], self.n_layers, 2, self.hidden_dim)
        gammas = out[:, :, 0, :]
        betas = out[:, :, 1, :]
        return [(gammas[:, i, :], betas[:, i, :]) for i in range(self.n_layers)]


class HyperLoRAConditioner(nn.Module):
    """Low-rank weight update conditioned on z (stronger than FiLM).

    W_eff = W + sum_r s_r(z) * A_r B_r^T

    We DO NOT materialize ΔW; instead compute (ΔW)x efficiently:
      (ΔW)x = sum_r s_r * A_r * (B_r · x)
    """

    def __init__(
        self,
        z_dim: int,
        in_dim: int,
        out_dim: int,
        rank: int = 4,
        width: int = 128,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.rank = int(rank)

        self.A = nn.Parameter(torch.randn(self.rank, self.out_dim) * 0.02)
        self.B = nn.Parameter(torch.randn(self.rank, self.in_dim) * 0.02)

        self.hyper = nn.Sequential(
            nn.Linear(self.z_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, self.rank),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[-1] != self.z_dim:
            raise ValueError(f"Expected z with shape (N,{self.z_dim}), got {tuple(z.shape)}")
        return self.hyper(z)

    def delta_output(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Compute (ΔW)x without building ΔW.

        x: (N, in_dim)
        s: (N, rank)
        returns: (N, out_dim)
        """
        if x.ndim != 2 or x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected x with shape (N,{self.in_dim}), got {tuple(x.shape)}")
        if s.ndim != 2 or s.shape[-1] != self.rank:
            raise ValueError(f"Expected s with shape (N,{self.rank}), got {tuple(s.shape)}")

        proj = x @ self.B.T          # (N, rank)
        weighted = proj * s          # (N, rank)
        return weighted @ self.A     # (N, out_dim)
