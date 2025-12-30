from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from config.toy_oscillator_param_config import ToyOscillatorParamConfig
from model.toy_oscillator_param_pinn import ToyOscillatorParamPINN
from trainer.toy_oscillator_param_trainer import train_toy_oscillator_param, grad
from utils.analytic_damped_oscillator import analytic_solution
from utils.plotting import plot_prediction, plot_residual
from utils.repro import get_device, set_seed
from utils.sampling import UniformSampler


def main():
    cfg = ToyOscillatorParamConfig()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    if cfg.use_float64:
        torch.set_default_dtype(torch.float64)

    model = ToyOscillatorParamPINN(
        t0=cfg.t0, t1=cfg.t1,
        gamma_min=cfg.gamma_min, gamma_max=cfg.gamma_max,
        omega_min=cfg.omega_min, omega_max=cfg.omega_max,
        x0=cfg.x0, v0=cfg.v0,
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
        activation=cfg.activation,
        time_fourier_dim=cfg.time_fourier_dim,
        time_fourier_sigma=cfg.time_fourier_sigma,
        use_residual=cfg.use_residual,
        cond_hidden=cfg.cond_hidden,
        cond_depth=cfg.cond_depth,
        film_scale=cfg.film_scale,
        gate_type=cfg.gate_type,
        gate_tau=cfg.gate_tau,
    ).to(device)

    sampler = UniformSampler(
        t0=cfg.t0, t1=cfg.t1,
        gamma_min=cfg.gamma_min, gamma_max=cfg.gamma_max,
        omega_min=cfg.omega_min, omega_max=cfg.omega_max,
        device=device,
        dtype=torch.get_default_dtype()
    )

    Path("outputs").mkdir(parents=True, exist_ok=True)

    _ = train_toy_oscillator_param(
        model=model,
        sampler=sampler,
        adam_steps=cfg.adam_steps,
        adam_lr=cfg.adam_lr,
        grad_clip=cfg.grad_clip,
        use_lbfgs=cfg.use_lbfgs,
        lbfgs_steps=cfg.lbfgs_steps,
        lbfgs_max_iter=cfg.lbfgs_max_iter,
        lbfgs_lr=cfg.lbfgs_lr,
        batch_size=cfg.batch_size,
        w_abs=cfg.w_abs,
        w_rel=cfg.w_rel,
        rel_eps=cfg.rel_eps,
        use_anchor=cfg.use_anchor,
        anchor_points=cfg.anchor_points,
        anchor_tmax=cfg.anchor_tmax,
        w_anchor=cfg.w_anchor,
        x0=cfg.x0,
        v0=cfg.v0,
        use_bl=cfg.use_bl,
        bl_points=cfg.bl_points,
        bl_tmax=cfg.bl_tmax,
        w_bl=cfg.w_bl,
        t0=cfg.t0,
        use_energy=cfg.use_energy,
        w_energy=cfg.w_energy,
        use_rar=cfg.use_rar,
        rar_warmup=cfg.rar_warmup,
        rar_prob=cfg.rar_prob,
        rar_candidates=cfg.rar_candidates,
        rar_tmax=cfg.rar_tmax,
        log_every=cfg.log_every,
        save_path=cfg.save_path
    )

    # Evaluate on one parameter instance
    gamma_test = 0.15
    omega_test = 2.5

    t = torch.linspace(cfg.t0, cfg.t1, 1500, device=device).reshape(-1, 1)
    gamma = torch.full_like(t, gamma_test)
    omega = torch.full_like(t, omega_test)

    # BC check
    t0 = torch.tensor([[cfg.t0]], device=device, dtype=torch.get_default_dtype(), requires_grad=True)
    g0 = torch.tensor([[gamma_test]], device=device, dtype=torch.get_default_dtype())
    w0 = torch.tensor([[omega_test]], device=device, dtype=torch.get_default_dtype())
    x0_pred, _, _, _ = model(t0, g0, w0)
    x0_t = grad(x0_pred, t0, create_graph=True)
    print("BC check:")
    print(f"  x(0) pred  = {float(x0_pred.detach().cpu()):.16e}  target = {cfg.x0:.16e}")
    print(f"  x'(0) pred = {float(x0_t.detach().cpu()):.16e} target = {cfg.v0:.16e}")

    # Residual on dense grid
    t_req = t.clone().requires_grad_(True)
    x_pred, _, _, _ = model(t_req, gamma, omega)
    x_t = grad(x_pred, t_req, create_graph=True)
    x_tt = grad(x_t, t_req, create_graph=True)
    res_x = (x_tt + 2.0*gamma*x_t + (omega**2)*x_pred).detach().cpu().numpy().reshape(-1)

    with torch.no_grad():
        x_true = analytic_solution(t, gamma, omega, x0=cfg.x0, v0=cfg.v0).cpu().numpy().reshape(-1)
        x_hat  = x_pred.detach().cpu().numpy().reshape(-1)

    t_np = t.detach().cpu().numpy().reshape(-1)
    plot_prediction(
        t=t_np,
        x_true=x_true,
        x_pred=x_hat,
        title=f"Toy PINN (Param Weak-Ansatz): γ={gamma_test}, ω={omega_test}",
        out_path="outputs/toy_oscillator_param_prediction.png"
    )
    plot_residual(
        t=t_np,
        res=res_x,
        title="Toy PINN (Param Weak-Ansatz): pointwise residual",
        out_path="outputs/toy_oscillator_param_residual.png"
    )

    rel_l2 = np.linalg.norm(x_hat - x_true) / (np.linalg.norm(x_true) + 1e-15)
    print(f"relative L2 error on test params: {rel_l2:.3e}")
    print("saved:")
    print("  outputs/toy_oscillator_param_prediction.png")
    print("  outputs/toy_oscillator_param_residual.png")
    print(f"  checkpoint: {cfg.save_path}")


if __name__ == "__main__":
    main()
