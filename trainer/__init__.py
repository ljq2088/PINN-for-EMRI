from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from model.toy_oscillator_pinn import ToyOscillatorPINN
from utils.sampling import UniformSampler

@dataclass
class TrainState:
    step: int
    loss_total: float
    loss_pde: float
    loss_bc: float

def _grad(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    (du_dx,) = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )
    return du_dx

@torch.no_grad()
def save_checkpoint(model: ToyOscillatorPINN, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def train_toy_oscillator_pinn(
    model: ToyOscillatorPINN,
    sampler: UniformSampler,
    x0: float,
    v0: float,
    steps: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    bc_batch_size: int,
    w_pde: float,
    w_bc: float,
    rel_eps: float,
    log_every: int,
    ckpt_path: str | Path | None = None,
):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    x0_t = torch.tensor(x0, device=sampler.device, dtype=sampler.dtype)
    v0_t = torch.tensor(v0, device=sampler.device, dtype=sampler.dtype)

    history = []

    for step in range(1, steps+1):
        # PDE residual
        t, gamma, omega = sampler.sample(batch_size)
        t.requires_grad_(True)
        x = model(t, gamma, omega)
        dx = _grad(x, t)
        d2x = _grad(dx, t)
        res = d2x + 2.0*gamma*dx + (omega**2)*x
        denom = d2x.abs() + (2.0*gamma*dx).abs() + ((omega**2)*x).abs() + rel_eps
        rel = res/denom
        loss_pde = torch.mean(rel**2)

        # BC
        t0, gamma0, omega0 = sampler.sample_bc(bc_batch_size)
        t0.requires_grad_(True)
        x_b = model(t0, gamma0, omega0)
        dx_b = _grad(x_b, t0)
        loss_bc = torch.mean((x_b.squeeze(-1)-x0_t)**2) + torch.mean((dx_b.squeeze(-1)-v0_t)**2)

        loss = w_pde*loss_pde + w_bc*loss_bc

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == 1 or step == steps:
            st = TrainState(step, float(loss.detach().cpu()), float(loss_pde.detach().cpu()), float(loss_bc.detach().cpu()))
            history.append(st)
            print(f"[step {st.step:6d}] loss_total={st.loss_total:.3e} (pde={st.loss_pde:.3e}, bc={st.loss_bc:.3e})", flush=True)

    if ckpt_path is not None:
        save_checkpoint(model, ckpt_path)

    return history
