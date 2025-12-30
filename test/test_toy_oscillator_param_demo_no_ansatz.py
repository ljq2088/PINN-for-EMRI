from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent))
from model.toy_oscillator_pinn_noansatz import ToyOscillatorPINNNoAnsatz
from physics.toy_oscillator import analytic_solution, OscillatorIC


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="saved_models/toy_osc_noansatz/toy_oscillator_noansatz.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = ckpt["cfg"]
    ic: OscillatorIC = ckpt["ic"]

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
        hyperlora_rank=0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    t = np.linspace(0.0, cfg.t_max, 1000)
    tt = torch.tensor(t, dtype=torch.float32, device=device)[:, None]

    gammas = np.linspace(cfg.param_range.gamma_min, cfg.param_range.gamma_max, 4)
    omegas = np.linspace(cfg.param_range.omega_min, cfg.param_range.omega_max, 4)

    rel_errors = []
    for g in gammas:
        for w in omegas:
            gg = torch.full_like(tt, float(g))
            ww = torch.full_like(tt, float(w))
            with torch.no_grad():
                u_pred = model(tt, gg, ww).squeeze(-1).cpu().numpy()
            u_true = analytic_solution(t, gamma=float(g), omega=float(w), u0=ic.u0, v0=ic.v0)
            num = np.linalg.norm(u_pred - u_true)
            den = np.linalg.norm(u_true) + 1e-12
            rel_errors.append(num / den)

    rel_errors = np.array(rel_errors)
    print(f"Relative L2 error: mean={rel_errors.mean():.3e}, max={rel_errors.max():.3e}")


if __name__ == "__main__":
    main()
