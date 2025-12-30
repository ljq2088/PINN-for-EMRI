from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from model.toy_oscillator_pinn import ToyOscillatorPINN
from utils.sampling import UniformSampler
from tqdm import tqdm  
import csv  
import matplotlib.pyplot as plt  
@dataclass
class TrainState:
    step: int
    loss_total: float
    loss_pde: float
    loss_bc: float
class EarlyStopping:
    """早停机制，当验证损失不再下降时停止训练"""
    def __init__(self, patience: int = 100, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, loss: float) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

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


# ========== 新增：保存loss数据到CSV ==========
def save_loss_history(history: list[TrainState], path: str | Path) -> None:
    """将训练历史保存为CSV文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss_total', 'loss_pde', 'loss_bc'])
        for state in history:
            writer.writerow([state.step, state.loss_total, state.loss_pde, state.loss_bc])

# ========== 新增：绘制loss曲线 ==========
def plot_loss_curve(history: list[TrainState], path: str | Path) -> None:
    """绘制并保存loss曲线"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    steps = [s.step for s in history]
    loss_total = [s.loss_total for s in history]
    loss_pde = [s.loss_pde for s in history]
    loss_bc = [s.loss_bc for s in history]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 总损失
    axes[0].semilogy(steps, loss_total, 'b-', linewidth=1.5, label='Total Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # PDE损失
    axes[1].semilogy(steps, loss_pde, 'r-', linewidth=1.5, label='PDE Loss')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('PDE Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # BC损失
    axes[2].semilogy(steps, loss_bc, 'g-', linewidth=1.5, label='BC Loss')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('BC (Initial Condition) Loss')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ========== 新增：综合绘制所有loss在一张图 ==========
def plot_loss_curve_combined(history: list[TrainState], path: str | Path) -> None:
    """绘制所有loss曲线在同一张图上"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    steps = [s.step for s in history]
    loss_total = [s.loss_total for s in history]
    loss_pde = [s.loss_pde for s in history]
    loss_bc = [s.loss_bc for s in history]
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(steps, loss_total, 'b-', linewidth=2, label='Total Loss')
    plt.semilogy(steps, loss_pde, 'r--', linewidth=1.5, label='PDE Loss')
    plt.semilogy(steps, loss_bc, 'g-.', linewidth=1.5, label='BC Loss')
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
# ================================================
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
    early_stop_patience: int = 200,      # 早停耐心值
    early_stop_min_delta: float = 1e-7,  # 早停最小变化
    ckpt_path: str | Path | None = None,
):
    model.train()
    early_stopper = EarlyStopping(patience=early_stop_patience, min_delta=early_stop_min_delta)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    x0_t = torch.tensor(x0, device=sampler.device, dtype=sampler.dtype)
    v0_t = torch.tensor(v0, device=sampler.device, dtype=sampler.dtype)

    history = []
    pbar = tqdm(range(1, steps + 1), desc="Training", unit="step")
    for step in pbar:
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
        st = TrainState(step, float(loss.detach().cpu()), float(loss_pde.detach().cpu()), float(loss_bc.detach().cpu()))
        history.append(st)
        if step % log_every == 0 or step == 1 or step == steps:
            
            
            pbar.set_postfix({
            'loss': f'{st.loss_total:.3e}',
            'pde': f'{st.loss_pde:.3e}',
            'bc': f'{st.loss_bc:.3e}'
        })
            # print(f"[step {st.step:6d}] loss_total={st.loss_total:.3e} (pde={st.loss_pde:.3e}, bc={st.loss_bc:.3e})", flush=True)
        if early_stopper(st.loss_total):
            pbar.write(f"[Early Stopping] 训练在第 {step} 步提前终止，最佳损失: {early_stopper.best_loss:.3e}")
            break
        if ckpt_path is not None and step % log_every == 0:
            save_checkpoint(model, ckpt_path)
            pbar.write(f"[Checkpoint] Step {step}: 模型已保存至 {ckpt_path}")
    if ckpt_path is not None:
        save_checkpoint(model, ckpt_path)
        ckpt_path = Path(ckpt_path)
        base_name = ckpt_path.stem  # 获取文件名（不含扩展名），如 "toy_oscillator"
        save_dir = ckpt_path.parent
        
        # 保存CSV数据
        csv_path = save_dir / f"{base_name}_loss_history.csv"
        save_loss_history(history, csv_path)
        pbar.write(f"[Loss Data] 损失数据已保存至 {csv_path}")
        
        # 保存分离的loss曲线图
        plot_path = save_dir / f"{base_name}_loss_curves.png"
        plot_loss_curve(history, plot_path)
        pbar.write(f"[Loss Plot] 分离损失曲线已保存至 {plot_path}")
        
        # 保存合并的loss曲线图
        plot_combined_path = save_dir / f"{base_name}_loss_combined.png"
        plot_loss_curve_combined(history, plot_combined_path)
        pbar.write(f"[Loss Plot] 合并损失曲线已保存至 {plot_combined_path}")

    return history
