from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent))
from model.toy_oscillator_pinn_noansatz import ToyOscillatorPINNNoAnsatz
from physics.toy_oscillator import OscillatorIC, analytic_solution
from trainer.toy_oscillator_trainer_noansatz import TrainConfig, ToyOscillatorTrainerNoAnsatz
from utils.plotting import plot_curves
from utils.sampling import ParamRange


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="saved_models/toy_osc_noansatz")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--t_max", type=float, default=20.0)
    parser.add_argument("--adam_steps", type=int, default=6000)
    parser.add_argument("--lbfgs_steps", type=int, default=200)
    parser.add_argument("--hyperlora_rank", type=int, default=0)  # 可选：>0 更强调制
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    pr = ParamRange(gamma_min=0.05, gamma_max=0.4, omega_min=0.5, omega_max=3.0)
    cfg = TrainConfig(
        t_max=args.t_max,
        param_range=pr,
        n_collocation=4096,
        n_bc=512,
        n_near=1024,
        t_eps=0.5,
        w_pde=1.0,
        w_bc=80.0,
        w_near=20.0,
        use_relative_residual=True,
        taylor_order=3,
        lr=2e-3,
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        rar_every=600,
        rar_pool=8192,
        rar_topk=1024,
        grad_clip=1.0,
        seed=123,
    )

    ic = OscillatorIC(u0=1.0, v0=0.0)

    model = ToyOscillatorPINNNoAnsatz(
        t_max=cfg.t_max,
        time_encoding="fourier",
        n_time_features=32,
        fourier_sigma=3.0,
        learnable_fourier=True,
        hidden_dim=128,
        n_layers=6,
        act="silu",
        film_width=128,
        param_embed_width=64,
        hyperlora_rank=int(args.hyperlora_rank),
        hyperlora_width=128,
    )

    trainer = ToyOscillatorTrainerNoAnsatz(model=model, cfg=cfg, ic=ic, device=device)
    metrics = trainer.train(verbose=True)
    print("Final metrics:", metrics)

    torch.save(
        {"model_state": model.state_dict(), "cfg": cfg, "ic": ic},
        outdir / "toy_oscillator_noansatz.pt",
    )

    # visualize
    t = np.linspace(0.0, cfg.t_max, 2000)
    test_params = [(0.08, 2.5), (0.25, 2.5), (0.35, 1.2)]

    model.eval()
    for (g, w) in test_params:
        tt = torch.tensor(t, dtype=torch.float32, device=device)[:, None]
        gg = torch.full_like(tt, float(g))
        ww = torch.full_like(tt, float(w))
        with torch.no_grad():
            u_pred = model(tt, gg, ww).squeeze(-1).cpu().numpy()
        u_true = analytic_solution(t, gamma=g, omega=w, u0=ic.u0, v0=ic.v0)
        plot_curves(
            t,
            curves=[("PINN(no-ansatz)", u_pred), ("analytic", u_true)],
            title=f"gamma={g:.3f}, omega={w:.3f}",
            outfile=str(f"outputs/curve_g{g:.3f}_w{w:.3f}.png"),
        )

    print(f"Saved to: {outdir}")


if __name__ == "__main__":
    main()
