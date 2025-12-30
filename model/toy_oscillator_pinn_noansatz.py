from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .encodings import GaussianFourierFeatures, IdentityEncoding, PositionalEncoding
from .conditioners import FiLMConditioner, HyperLoRAConditioner


def _make_mlp(in_dim: int, hidden_dim: int, n_layers: int, out_dim: int, act: Literal["silu", "tanh"] = "silu"):
    if n_layers < 2:
        raise ValueError("n_layers must be >= 2")
    if act == "silu":
        activation = nn.SiLU()
    elif act == "tanh":
        activation = nn.Tanh()
    else:
        raise ValueError(act)

    layers = [nn.Linear(in_dim, hidden_dim), activation]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), activation]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class ToyOscillatorPINNNoAnsatz(nn.Module):
    """Parameter-conditioned PINN for damped oscillator, WITHOUT analytic ansatz.

    Inputs:
      t: (N,1)
      gamma: (N,1)
      omega: (N,1)
    Output:
      u(t; gamma, omega): (N,1)

    Design:
      - Time encoding: Fourier/positional features.
      - Conditioning: FiLM (default) and optional HyperLoRA (stronger than FiLM).
      - No hand-crafted analytic solution form.
    """

    def __init__(
        self,
        t_max: float,
        time_encoding: Literal["fourier", "positional", "none"] = "fourier",
        n_time_features: int = 32,
        fourier_sigma: float = 3.0,
        learnable_fourier: bool = True,
        hidden_dim: int = 128,
        n_layers: int = 6,
        act: Literal["silu", "tanh"] = "silu",
        film_width: int = 128,
        param_embed_width: int = 64,
        hyperlora_rank: int = 0,
        hyperlora_width: int = 128,
    ) -> None:
        super().__init__()
        self.t_max = float(t_max)

        # ---- time encoding ----
        if time_encoding == "fourier":
            self.time_enc = GaussianFourierFeatures(
                n_features=int(n_time_features),
                sigma=float(fourier_sigma),
                learnable=bool(learnable_fourier),
                seed=1234,
            )
            time_dim = 2 * int(n_time_features)
        elif time_encoding == "positional":
            self.time_enc = PositionalEncoding(n_freqs=int(n_time_features // 2), max_freq=64.0, include_input=True)
            time_dim = 1 + 2 * int(n_time_features // 2)
        elif time_encoding == "none":
            self.time_enc = IdentityEncoding()
            time_dim = 1
        else:
            raise ValueError(f"Unknown time_encoding={time_encoding}")

        # ---- parameter embedding (gamma, omega) ----
        self.param_embed = _make_mlp(
            in_dim=2, hidden_dim=param_embed_width, n_layers=3, out_dim=param_embed_width, act=act
        )
        self.z_dim = int(param_embed_width)

        # ---- trunk network for time ----
        if n_layers < 3:
            raise ValueError("n_layers should be >= 3 for this implementation.")
        self.in_proj = nn.Linear(time_dim, hidden_dim)
        self.hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)])
        self.out_proj = nn.Linear(hidden_dim, 1)
        self.n_mod_layers = 1 + len(self.hidden)

        if act == "silu":
            self.act = nn.SiLU()
        elif act == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(act)

        # FiLM modulation per layer
        self.film = FiLMConditioner(
            z_dim=self.z_dim,
            hidden_dim=hidden_dim,
            n_layers=self.n_mod_layers,
            width=film_width,
            act="silu",
        )

        # Optional HyperLoRA modulation
        self.hyperlora_rank = int(hyperlora_rank)
        if self.hyperlora_rank > 0:
            self.hl_in = HyperLoRAConditioner(
                z_dim=self.z_dim,
                in_dim=time_dim,
                out_dim=hidden_dim,
                rank=self.hyperlora_rank,
                width=int(hyperlora_width),
            )
            self.hl_hidden = nn.ModuleList(
                [
                    HyperLoRAConditioner(
                        z_dim=self.z_dim,
                        in_dim=hidden_dim,
                        out_dim=hidden_dim,
                        rank=self.hyperlora_rank,
                        width=int(hyperlora_width),
                    )
                    for _ in range(len(self.hidden))
                ]
            )
        else:
            self.hl_in = None
            self.hl_hidden = None

    def forward(self, t: torch.Tensor, gamma: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        if gamma.ndim == 1:
            gamma = gamma[:, None]
        if omega.ndim == 1:
            omega = omega[:, None]

        # normalize time for encoding stability
        t01 = t / self.t_max
        t_centered = t01 - 0.5

        xt = self.time_enc(t_centered)              # (N, time_dim)
        z_raw = torch.cat([gamma, omega], dim=-1)   # (N,2)
        z = self.param_embed(z_raw)                 # (N, z_dim)

        gammas_betas = self.film(z)

        # layer 0
        h = self.in_proj(xt)
        if self.hl_in is not None:
            s0 = self.hl_in(z)
            h = h + self.hl_in.delta_output(xt, s0)
        h = self.act(h)
        g0, b0 = gammas_betas[0]
        h = (1.0 + g0) * h + b0

        # hidden layers
        for i, layer in enumerate(self.hidden, start=1):
            base = layer(h)
            if self.hl_hidden is not None:
                si = self.hl_hidden[i - 1](z)
                base = base + self.hl_hidden[i - 1].delta_output(h, si)
            h = self.act(base)
            gi, bi = gammas_betas[i]
            h = (1.0 + gi) * h + bi

        return self.out_proj(h)
