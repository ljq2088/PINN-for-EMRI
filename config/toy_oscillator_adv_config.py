from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ToyOscillatorAdvConfig:
    # domain
    t0: float = 0.0
    t1: float = 20.0

    # parameter ranges (guarantee underdamped: omega_min > gamma_max)
    gamma_min: float = 0.05
    gamma_max: float = 0.50
    omega_min: float = 1.00
    omega_max: float = 4.00

    # IC for x(t)
    x0: float = 1.0
    v0: float = 0.0

    # model
    hidden_dim: int = 128
    num_blocks: int = 4
    activation: str = "silu"

    # Fourier features for s in [0,1]
    time_fourier_dim: int = 64
    time_fourier_sigma: float = 2.0

    # FiLM conditioning
    cond_hidden: int = 64
    cond_depth: int = 2
    use_residual: bool = True
    film_scale: float = 0.01  # safer default

    # training
    device: str = "cuda"
    seed: int = 1234
    use_float64: bool = True

    # Adam
    adam_steps: int = 7000
    adam_lr: float = 1e-3
    grad_clip: float = 1.0

    # LBFGS
    use_lbfgs: bool = True
    lbfgs_steps: int = 800
    lbfgs_max_iter: int = 30
    lbfgs_lr: float = 1.0

    batch_size: int = 2048

    # losses
    rel_eps: float = 1e-12
    w_abs: float = 1.0
    w_rel: float = 0.0  # keep 0 at first; can try 0.05 later

    # anchor loss (locks phase with very few points)
    use_anchor: bool = True
    anchor_points: int = 64
    anchor_tmax: float = 8.0   # focus anchors where error is largest
    w_anchor: float = 5e-2

    # boundary layer loss (behavior near boundary, not only at boundary point)
    use_bl: bool = True
    bl_tmax: float = 1.5       # boundary layer thickness near t0
    bl_points: int = 256
    w_bl: float = 2e-1

    # optional energy consistency loss (weak constraint)
    use_energy: bool = True
    w_energy: float = 1e-3

    # time-window RAR
    use_rar: bool = True
    rar_warmup: int = 1200
    rar_prob: float = 0.40
    rar_candidates: int = 8192
    rar_tmax: float = 8.0      # only refine early-time region

    log_every: int = 300
    save_path: str = "outputs/toy_oscillator_adv_full.pt"
