from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ToyOscillatorPINNConfig:
    # domain
    t0: float = 0.0
    t1: float = 20.0

    # parameter ranges for parametric PINN
    gamma_min: float = 0.05
    gamma_max: float = 0.50
    omega_min: float = 1.0
    omega_max: float = 4.0

    # IC
    x0: float = 1.0
    v0: float = 0.0

    # model
    hidden_dim: int = 64
    num_layers: int = 5
    activation: str = "tanh"      # {"tanh","silu"}
    use_fourier: bool = True
    fourier_dim: int = 32
    fourier_sigma: float = 2.0
    use_residual: bool = True

    # train
    device: str = "cuda"
    seed: int = 1234
    steps: int = 1500
    lr: float = 2e-3
    weight_decay: float = 0.0

    batch_size: int = 512
    bc_batch_size: int = 128

    w_pde: float = 1.0
    w_bc: float = 50.0
    rel_eps: float = 1e-6

    log_every: int = 200
    save_path: str = "saved_models/toy_oscillator_pinn.pt"
