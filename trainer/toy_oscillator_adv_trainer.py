from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch

from model.toy_oscillator_adv_pinn import ToyOscillatorAdvPINN
from utils.sampling import UniformSampler
from utils.analytic_damped_oscillator import analytic_solution

def grad(u: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    (du_dx,) = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=create_graph,
        retain_graph=True,
    )
    return du_dx

@dataclass
class TrainLog:
    step: int
    loss: float
    loss_abs: float
    loss_anchor: float
    loss_bl: float
    loss_energy: float

@torch.no_grad()
def save_ckpt(model: ToyOscillatorAdvPINN, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def residual_x(model: ToyOscillatorAdvPINN, t: torch.Tensor, gamma: torch.Tensor, omega: torch.Tensor):
    """
    original ODE residual:
      r = x'' + 2γ x' + ω^2 x
    """
    t = t.requires_grad_(True)
    x, _, _, _, _ = model(t, gamma, omega)
    x_t = grad(x, t, create_graph=True)
    x_tt = grad(x_t, t, create_graph=True)
    r = x_tt + 2.0*gamma*x_t + (omega**2)*x
    return r, x, x_t, x_tt

def boundary_layer_series(t: torch.Tensor, gamma: torch.Tensor, omega: torch.Tensor, x0: float, v0: float):
    """
    Taylor expansion around t0=0 (assumed), for small t:
      x(t) = x0 + v0 t + 0.5 x2 t^2 + (1/6) x3 t^3
    where:
      x2 = x''(0) = -2γ v0 - ω^2 x0
      x3 = x'''(0) = -2γ x2 - ω^2 v0
    """
    x0_t = torch.as_tensor(float(x0), dtype=t.dtype, device=t.device)
    v0_t = torch.as_tensor(float(v0), dtype=t.dtype, device=t.device)

    x2 = -2.0*gamma*v0_t - (omega**2)*x0_t
    x3 = -2.0*gamma*x2 - (omega**2)*v0_t

    return x0_t + v0_t*t + 0.5*x2*(t**2) + (1.0/6.0)*x3*(t**3)

def energy_consistency_loss(t: torch.Tensor, x: torch.Tensor, x_t: torch.Tensor, omega: torch.Tensor, gamma: torch.Tensor):
    """
    E = 1/2 (x'^2 + ω^2 x^2)
    dE/dt = x' x'' + ω^2 x x'
    For the damped oscillator, exact identity:
      dE/dt + 2γ x'^2 = 0
    We enforce: (dE/dt + 2γ x'^2)^2
    """
    t = t.requires_grad_(True)
    # x and x_t must depend on t; x_t is already grad(x,t)
    # compute x_tt from x_t
    x_tt = grad(x_t, t, create_graph=True)
    dE = x_t * x_tt + (omega**2) * x * x_t
    resE = dE + 2.0*gamma*(x_t**2)
    return torch.mean(resE**2)

def sample_batch_rar(model: ToyOscillatorAdvPINN, sampler: UniformSampler, batch_size: int,
                     use_rar: bool, rar_prob: float, rar_candidates: int, rar_tmax: float):
    if (not use_rar) or (torch.rand(()) > rar_prob):
        return sampler.sample(batch_size)

    t_c, g_c, w_c = sampler.sample(rar_candidates, t_low=sampler.t0, t_high=rar_tmax)
    r, _, _, _ = residual_x(model, t_c, g_c, w_c)
    score = r.detach().abs().squeeze(-1)
    topk = torch.topk(score, k=batch_size, largest=True).indices
    return t_c[topk].detach(), g_c[topk].detach(), w_c[topk].detach()

def train_toy_oscillator_adv(
    model: ToyOscillatorAdvPINN,
    sampler: UniformSampler,
    adam_steps: int,
    adam_lr: float,
    grad_clip: float,
    use_lbfgs: bool,
    lbfgs_steps: int,
    lbfgs_max_iter: int,
    lbfgs_lr: float,
    batch_size: int,
    # weights
    w_abs: float,
    w_rel: float,
    rel_eps: float,
    # anchor
    use_anchor: bool,
    anchor_points: int,
    anchor_tmax: float,
    w_anchor: float,
    x0: float,
    v0: float,
    # boundary layer
    use_bl: bool,
    bl_points: int,
    bl_tmax: float,
    w_bl: float,
    # energy
    use_energy: bool,
    w_energy: float,
    # RAR
    use_rar: bool,
    rar_warmup: int,
    rar_prob: float,
    rar_candidates: int,
    rar_tmax: float,
    log_every: int,
    save_path: str | Path | None = None,
):
    model.train()
    logs: list[TrainLog] = []

    opt = torch.optim.Adam(model.parameters(), lr=adam_lr)

    for step in range(1, adam_steps + 1):
        rar_on = use_rar and (step >= rar_warmup)

        # main collocation (RAR focuses early-time window)
        t, g, w = sample_batch_rar(model, sampler, batch_size, rar_on, rar_prob, rar_candidates, rar_tmax)

        r, x_pred, x_t, x_tt = residual_x(model, t, g, w)
        loss_abs = torch.mean(r**2)

        denom = (x_tt.abs() + (2.0*g*x_t).abs() + ((w**2)*x_pred).abs() + rel_eps)
        loss_rel = torch.mean((r/denom)**2)

        # anchor: few points with analytic x(t) to lock phase
        if use_anchor and anchor_points > 0:
            t_a, g_a, w_a = sampler.sample(anchor_points, t_low=sampler.t0, t_high=anchor_tmax)
            with torch.no_grad():
                x_true = analytic_solution(t_a, g_a, w_a, x0=x0, v0=v0)
            x_hat, _, _, _, _ = model(t_a, g_a, w_a)
            loss_anchor = torch.mean((x_hat - x_true)**2)
        else:
            loss_anchor = torch.zeros((), device=t.device, dtype=t.dtype)

        # boundary-layer behavior loss near t0: match ODE-derived Taylor series
        if use_bl and bl_points > 0:
            t_bl, g_bl, w_blp = sampler.sample(bl_points, t_low=sampler.t0, t_high=bl_tmax)
            x_hat_bl, _, _, _, _ = model(t_bl, g_bl, w_blp)
            x_ser = boundary_layer_series(t_bl, g_bl, w_blp, x0=x0, v0=v0)
            loss_bl = torch.mean((x_hat_bl - x_ser)**2)
        else:
            loss_bl = torch.zeros((), device=t.device, dtype=t.dtype)

        # energy consistency (weak)
        if use_energy:
            # reuse (t,x_pred,x_t) but needs t with grad; recompute on a smaller batch to save cost
            t_e, g_e, w_e = sampler.sample(min(512, batch_size))
            t_e = t_e.detach().requires_grad_(True)
            x_e, _, _, _, _ = model(t_e, g_e, w_e)
            x_e_t = grad(x_e, t_e, create_graph=True)
            loss_energy = energy_consistency_loss(t_e, x_e, x_e_t, w_e, g_e)
        else:
            loss_energy = torch.zeros((), device=t.device, dtype=t.dtype)

        loss = (
            w_abs*loss_abs
            + w_rel*loss_rel
            + w_anchor*loss_anchor
            + w_bl*loss_bl
            + w_energy*loss_energy
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if step % log_every == 0 or step in {1, adam_steps}:
            logs.append(TrainLog(
                step=step,
                loss=float(loss.detach().cpu()),
                loss_abs=float(loss_abs.detach().cpu()),
                loss_anchor=float(loss_anchor.detach().cpu()),
                loss_bl=float(loss_bl.detach().cpu()),
                loss_energy=float(loss_energy.detach().cpu()),
            ))
            L = logs[-1]
            print(
                f"[Adam {step:6d}] loss={L.loss:.3e} | abs={L.loss_abs:.3e} "
                f"| anchor={L.loss_anchor:.3e} | bl={L.loss_bl:.3e} | E={L.loss_energy:.3e}",
                flush=True
            )

    # LBFGS refinement
    if use_lbfgs:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,
            line_search_fn="strong_wolfe",
        )

        for k in range(1, lbfgs_steps + 1):
            t, g, w = sampler.sample(batch_size)
            t = t.detach().requires_grad_(True)

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                r, x_pred, x_t, x_tt = residual_x(model, t, g, w)
                loss_abs = torch.mean(r**2)
                denom = (x_tt.abs() + (2.0*g*x_t).abs() + ((w**2)*x_pred).abs() + rel_eps)
                loss_rel = torch.mean((r/denom)**2)

                # keep anchor & boundary-layer during LBFGS too (small, but stabilizes phase)
                if use_anchor and anchor_points > 0:
                    t_a, g_a, w_a = sampler.sample(anchor_points, t_low=sampler.t0, t_high=anchor_tmax)
                    with torch.no_grad():
                        x_true = analytic_solution(t_a, g_a, w_a, x0=x0, v0=v0)
                    x_hat, _, _, _, _ = model(t_a, g_a, w_a)
                    loss_anchor = torch.mean((x_hat - x_true)**2)
                else:
                    loss_anchor = torch.zeros((), device=t.device, dtype=t.dtype)

                if use_bl and bl_points > 0:
                    t_bl, g_bl, w_blp = sampler.sample(bl_points, t_low=sampler.t0, t_high=bl_tmax)
                    x_hat_bl, _, _, _, _ = model(t_bl, g_bl, w_blp)
                    x_ser = boundary_layer_series(t_bl, g_bl, w_blp, x0=x0, v0=v0)
                    loss_bl = torch.mean((x_hat_bl - x_ser)**2)
                else:
                    loss_bl = torch.zeros((), device=t.device, dtype=t.dtype)

                loss = w_abs*loss_abs + w_rel*loss_rel + w_anchor*loss_anchor + w_bl*loss_bl
                loss.backward()
                return loss

            lv = lbfgs.step(closure)
            if k % 200 == 0 or k in {1, lbfgs_steps}:
                lvf = float(lv.detach().cpu()) if torch.is_tensor(lv) else float(lv)
                print(f"[LBFGS {k:6d}] loss={lvf:.3e}", flush=True)

    if save_path is not None:
        save_ckpt(model, save_path)

    return logs
