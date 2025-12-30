from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ToyOscillatorParamConfig:
    t0: float = 0.0
    t1: float = 20.0

    gamma_min: float = 0.05
    gamma_max: float = 0.50
    omega_min: float = 1.00
    omega_max: float = 4.00  # keep omega_min > gamma_max for underdamped if your analytic_solution assumes it

    x0: float = 1.0
    v0: float = 0.0

    # model
    hidden_dim: int = 192
    num_blocks: int = 5
    activation: str = "silu"
    time_fourier_dim: int = 96
    time_fourier_sigma: float = 3.0

    cond_hidden: int = 64
    cond_depth: int = 2
    use_residual: bool = True
    film_scale: float = 0.01

    # weak ansatz gate
    gate_type: str = "s2"     # "s2" (recommended), or "exp2"
    gate_tau: float = 1.0

    # training
    device: str = "cuda"
    seed: int = 1234
    use_float64: bool = True

    adam_steps: int = 9000
    adam_lr: float = 1e-3
    grad_clip: float = 1.0

    use_lbfgs: bool = True
    lbfgs_steps: int = 1000
    lbfgs_max_iter: int = 30
    lbfgs_lr: float = 1.0

    batch_size: int = 2048

    rel_eps: float = 1e-12
    w_abs: float = 1.0
    w_rel: float = 0.0

    # anchor (optional but helpful for phase locking)
    use_anchor: bool = True
    anchor_points: int = 96
    anchor_tmax: float = 8.0
    w_anchor: float = 5e-2

    # boundary layer: enforce near-boundary behavior (your key point)
    use_bl: bool = True
    bl_tmax: float = 1.5
    bl_points: int = 256
    w_bl: float = 2e-1

    # energy consistency (very weak)
    use_energy: bool = True
    w_energy: float = 1e-3

    # RAR on early-time window
    use_rar: bool = True
    rar_warmup: int = 1200
    rar_prob: float = 0.40
    rar_candidates: int = 8192
    rar_tmax: float = 8.0

    log_every: int = 300
    save_path: str = "outputs/toy_oscillator_param_pinn.pt"
