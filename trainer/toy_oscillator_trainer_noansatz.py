from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
from torch.optim import Adam
from torch.optim.lbfgs import LBFGS
from tqdm import tqdm

from physics.toy_oscillator import (
    OscillatorIC,
    oscillator_residual,
    relative_residual,
    taylor_u_near0,
)
from utils.sampling import ParamRange, sample_exact_zero, sample_near_zero, sample_uniform


@dataclass
class TrainConfig:
    t_max: float = 20.0
    param_range: ParamRange = field(
        default_factory=lambda: ParamRange(gamma_min=0.05, gamma_max=0.4, omega_min=0.5, omega_max=3.0)
    )

    # sampling
    n_collocation: int = 4096
    n_bc: int = 512
    n_near: int = 1024
    t_eps: float = 0.3

    # losses
    w_pde: float = 1.0
    w_bc: float = 80.0
    w_near: float = 20.0
    use_relative_residual: bool = True
    taylor_order: int = 3

    # optimizer
    lr: float = 2e-3
    adam_steps: int = 8000
    lbfgs_steps: int = 300

    # RAR
    rar_every: int = 500
    rar_pool: int = 8192
    rar_topk: int = 1024

    # misc
    grad_clip: float = 1.0
    seed: int = 123


class ToyOscillatorTrainerNoAnsatz:
    def __init__(self, model: torch.nn.Module, cfg: TrainConfig, ic: OscillatorIC, device: torch.device) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.ic = ic
        self.device = device
        torch.manual_seed(int(cfg.seed))
        self.rar_buffer = None  # (t,g,w)

    def _loss_components(self) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        pr = cfg.param_range

        # collocation
        t_f, g_f, w_f = sample_uniform(cfg.n_collocation, cfg.t_max, pr, self.device)
        if self.rar_buffer is not None:
            t_r, g_r, w_r = self.rar_buffer
            t_f = torch.cat([t_f, t_r], dim=0)
            g_f = torch.cat([g_f, g_r], dim=0)
            w_f = torch.cat([w_f, w_r], dim=0)

        # boundary t=0
        t0, g0, w0 = sample_exact_zero(cfg.n_bc, pr, self.device)

        # near-boundary
        tn, gn, wn = sample_near_zero(cfg.n_near, cfg.t_eps, pr, self.device)

        # PDE residual (t must require grad)
        t_f_req = t_f.clone().requires_grad_(True)
        u_f = self.model(t_f_req, g_f, w_f)
        r_f, u_t_f, u_tt_f = oscillator_residual(u_f, t_f_req, g_f, w_f)
        r_used = relative_residual(r_f, u_f, u_t_f, u_tt_f, g_f, w_f) if cfg.use_relative_residual else r_f
        loss_pde = torch.mean(r_used ** 2)

        # IC loss: u(0)=u0, u'(0)=v0
        t0_req = t0.clone().requires_grad_(True)
        u0_pred = self.model(t0_req, g0, w0)
        u0_t = torch.autograd.grad(
            outputs=u0_pred,
            inputs=t0_req,
            grad_outputs=torch.ones_like(u0_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        u0_true = torch.full_like(u0_pred, float(self.ic.u0))
        v0_true = torch.full_like(u0_t, float(self.ic.v0))
        loss_bc = torch.mean((u0_pred - u0_true) ** 2) + torch.mean((u0_t - v0_true) ** 2)

        # near-boundary Taylor constraint (value + slope)
        tn_req = tn.clone().requires_grad_(True)
        un_pred = self.model(tn_req, gn, wn)
        un_true = taylor_u_near0(tn_req, gn, wn, self.ic, order=cfg.taylor_order)

        un_t = torch.autograd.grad(
            outputs=un_pred,
            inputs=tn_req,
            grad_outputs=torch.ones_like(un_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        un_true_t = torch.autograd.grad(
            outputs=un_true,
            inputs=tn_req,
            grad_outputs=torch.ones_like(un_true),
            create_graph=False,
            retain_graph=True,
            only_inputs=True,
        )[0]
        loss_near = torch.mean((un_pred - un_true) ** 2) + torch.mean((un_t - un_true_t) ** 2)

        loss_total = cfg.w_pde * loss_pde + cfg.w_bc * loss_bc + cfg.w_near * loss_near
        return {
            "loss": loss_total,
            "pde": loss_pde.detach(),
            "bc": loss_bc.detach(),
            "near": loss_near.detach(),
        }

    def _rar_update(self) -> None:
        cfg = self.cfg
        pr = cfg.param_range
        t_pool, g_pool, w_pool = sample_uniform(cfg.rar_pool, cfg.t_max, pr, self.device)

        t_pool_req = t_pool.clone().requires_grad_(True)
        u_pool = self.model(t_pool_req, g_pool, w_pool)
        r_pool, u_t_pool, u_tt_pool = oscillator_residual(u_pool, t_pool_req, g_pool, w_pool)
        r_used = relative_residual(r_pool, u_pool, u_t_pool, u_tt_pool, g_pool, w_pool) if cfg.use_relative_residual else r_pool

        scores = (r_used.detach() ** 2).squeeze(-1)
        topk = min(int(cfg.rar_topk), scores.numel())
        idx = torch.topk(scores, k=topk, largest=True).indices
        self.rar_buffer = (t_pool[idx], g_pool[idx], w_pool[idx])

    def train(self, verbose: bool = True) -> Dict[str, float]:
        cfg = self.cfg
        opt = Adam(self.model.parameters(), lr=cfg.lr)

        pbar = tqdm(range(cfg.adam_steps), disable=not verbose)
        for step in pbar:
            opt.zero_grad(set_to_none=True)
            comps = self._loss_components()
            comps["loss"].backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=cfg.grad_clip)
            opt.step()

            if (step + 1) % 50 == 0:
                pbar.set_postfix({k: float(v) for k, v in comps.items() if k != "loss"})

            if cfg.rar_every > 0 and (step + 1) % cfg.rar_every == 0:
                self._rar_update()

        if cfg.lbfgs_steps and cfg.lbfgs_steps > 0:
            lbfgs = LBFGS(self.model.parameters(), lr=1.0, max_iter=cfg.lbfgs_steps, line_search_fn="strong_wolfe")

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                comps = self._loss_components()
                comps["loss"].backward()
                return comps["loss"]

            lbfgs.step(closure)

        comps = self._loss_components()
        return {k: float(v) for k, v in comps.items() if k != "loss"}
