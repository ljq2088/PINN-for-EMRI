import math
from typing import Optional

import torch
import torch.nn as nn


class IdentityEncoding(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class GaussianFourierFeatures(nn.Module):
    """Gaussian random Fourier features for scalar inputs.

    Input: x with shape (N, 1) or (N,)
    Output: (N, 2 * n_features)

    phi(x) = [sin(2π x f_i), cos(2π x f_i)]_{i=1..n_features}
    """

    def __init__(
        self,
        n_features: int,
        sigma: float = 1.0,
        learnable: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        g = torch.Generator()
        g.manual_seed(int(seed))
        freqs = torch.randn(self.n_features, generator=g) * float(sigma)
        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[-1] != 1:
            raise ValueError(f"Expected x shape (N,1) or (N,), got {tuple(x.shape)}")
        angles = 2.0 * math.pi * x * self.freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class PositionalEncoding(nn.Module):
    """Deterministic positional encoding for scalar inputs.

    Uses frequency bands 2^k up to max_freq.
    """

    def __init__(
        self,
        n_freqs: int,
        max_freq: float = 32.0,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.include_input = bool(include_input)
        if n_freqs <= 0:
            self.register_buffer("freq_bands", torch.empty(0))
        else:
            fb = 2 ** torch.linspace(0, math.log2(float(max_freq)), int(n_freqs))
            self.register_buffer("freq_bands", fb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[-1] != 1:
            raise ValueError(f"Expected x shape (N,1) or (N,), got {tuple(x.shape)}")
        if self.freq_bands.numel() == 0:
            return x if self.include_input else torch.empty(x.shape[0], 0, device=x.device, dtype=x.dtype)

        xb = x * self.freq_bands[None, :] * math.pi
        feats = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)
        if self.include_input:
            feats = torch.cat([x, feats], dim=-1)
        return feats
