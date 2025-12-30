from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from config.toy_oscillator_pinn_config import ToyOscillatorPINNConfig
from model.toy_oscillator_pinn import ToyOscillatorPINN
from trainer.toy_oscillator_trainer import train_toy_oscillator_pinn
from utils.analytic_damped_oscillator import analytic_solution
from utils.plotting import plot_prediction, plot_residual
from utils.repro import get_device, set_seed
from utils.sampling import UniformSampler

def main():
    cfg = ToyOscillatorPINNConfig()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    model = ToyOscillatorPINN(
        t0=cfg.t0, t1=cfg.t1,
        gamma_min=cfg.gamma_min, gamma_max=cfg.gamma_max,
        omega_min=cfg.omega_min, omega_max=cfg.omega_max,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        activation=cfg.activation,
        use_fourier=cfg.use_fourier,
        fourier_dim=cfg.fourier_dim,
        fourier_sigma=cfg.fourier_sigma,
        use_residual=cfg.use_residual,
    ).to(device)

    sampler = UniformSampler(
        t0=cfg.t0, t1=cfg.t1,
        gamma_min=cfg.gamma_min, gamma_max=cfg.gamma_max,
        omega_min=cfg.omega_min, omega_max=cfg.omega_max,
        device=device,
        dtype=torch.float32
    )

    history = train_toy_oscillator_pinn(
        model=model,
        sampler=sampler,
        x0=cfg.x0,
        v0=cfg.v0,
        steps=cfg.steps,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        batch_size=cfg.batch_size,
        bc_batch_size=cfg.bc_batch_size,
        w_pde=cfg.w_pde,
        w_bc=cfg.w_bc,
        rel_eps=cfg.rel_eps,
        log_every=cfg.log_every,
        ckpt_path=cfg.save_path,
        early_stop_patience=100,       # 可根据需要调整
        early_stop_min_delta=1e-5,     # 可根据需要调整
    )

    last = history[-1]
    print("--- ToyOscillatorPINN demo finished ---")
    print(f"device: {device}")
    print(f"final step: {last.step}")
    print(f"loss_total={last.loss_total:.3e} | loss_pde={last.loss_pde:.3e} | loss_bc={last.loss_bc:.3e}")

    gamma_test = 0.15
    omega_test = 2.5

    t = torch.linspace(cfg.t0, cfg.t1, 1000, device=device).reshape(-1,1)
    gamma = torch.full_like(t, gamma_test)
    omega = torch.full_like(t, omega_test)

    t_req = t.clone().requires_grad_(True)
    x_pred = model(t_req, gamma, omega)

    (dx,)  = torch.autograd.grad(x_pred, t_req, grad_outputs=torch.ones_like(x_pred), create_graph=True)
    (d2x,) = torch.autograd.grad(dx, t_req, grad_outputs=torch.ones_like(dx), create_graph=True)
    res = (d2x + 2.0*gamma*dx + (omega**2)*x_pred).detach().cpu().numpy().reshape(-1)

    with torch.no_grad():
        x_true = analytic_solution(t, gamma, omega, x0=cfg.x0, v0=cfg.v0).cpu().numpy().reshape(-1)
        x_hat  = x_pred.detach().cpu().numpy().reshape(-1)

    t_np = t.detach().cpu().numpy().reshape(-1)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_prediction(
        t=t_np,
        x_true=x_true,
        x_pred=x_hat,
        title=f"Toy PINN: x''+2γx'+ω²x=0 | γ={gamma_test}, ω={omega_test}",
        out_path=out_dir / "toy_oscillator_prediction.png"
    )

    plot_residual(
        t=t_np,
        res=res,
        title="Toy PINN: pointwise ODE residual",
        out_path=out_dir / "toy_oscillator_residual.png"
    )

    rel_l2 = np.linalg.norm(x_hat-x_true)/(np.linalg.norm(x_true)+1e-12)
    print(f"relative L2 error on test params: {rel_l2:.3e}")
    print(f"plots saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
