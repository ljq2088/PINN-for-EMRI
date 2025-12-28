import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1) MLP backbone (standard PINN)
# -----------------------------
class MLsmP(nn.Module):
    def __init__(self, in_dim=3, hidden=128, depth=4, act=nn.Tanh):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(act())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(act())
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# 2) Parametric PINN for damped oscillator
#    x'' + 2γ x' + ω^2 x = 0
# -----------------------------
class DampedOscillatorPINN(nn.Module):
    def __init__(self, hidden=128, depth=4):
        super().__init__()
        self.mlp = MLP(in_dim=3, hidden=hidden, depth=depth, act=nn.Tanh)

    def forward(self, t, gamma, omega):
        # t, gamma, omega: shape (N, 1)
        inp = torch.cat([t, gamma, omega], dim=1)
        x = self.mlp(inp)
        return x


def gradients(y, x):
    """dy/dx with autograd; y and x are tensors with same batch size."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]


# -----------------------------
# 3) Training loop
# -----------------------------
def train_pinn(
    device="cuda" if torch.cuda.is_available() else "cpu",
    gamma_range=(0.05, 0.5),
    omega_range=(1.0, 10.0),
    t_max=10.0,
    n_param=64,       # batch size in (gamma, omega)
    n_colloc=256,     # collocation points per param batch
    steps=20000,
    lr=1e-3,
    lambda_ic=10.0,
    x0=1.0,
    v0=0.0,
    eps=1e-12
):
    model = DampedOscillatorPINN(hidden=128, depth=4).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for step in range(1, steps + 1):
        # ---- sample parameters (continuous inputs) ----
        gamma = torch.rand(n_param, 1, device=device) * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
        omega = torch.rand(n_param, 1, device=device) * (omega_range[1] - omega_range[0]) + omega_range[0]

        # ---- sample collocation times (continuous) ----
        # shape (n_param * n_colloc, 1)
        t = torch.rand(n_param * n_colloc, 1, device=device) * t_max
        t.requires_grad_(True)

        # broadcast gamma, omega to match collocation points
        gamma_c = gamma.repeat_interleave(n_colloc, dim=0)
        omega_c = omega.repeat_interleave(n_colloc, dim=0)

        # ---- forward ----
        x = model(t, gamma_c, omega_c)
        x_t = gradients(x, t)
        x_tt = gradients(x_t, t)

        # ODE residual
        r = x_tt + 2.0 * gamma_c * x_t + (omega_c ** 2) * x

        # relative residual loss (one robust version)
        denom = (x_tt**2 + (2.0*gamma_c*x_t)**2 + ((omega_c**2)*x)**2).detach() + eps
        loss_ode = torch.mean((r**2) / denom)

        # ---- initial condition loss (soft) ----
        t0 = torch.zeros(n_param, 1, device=device, requires_grad=True)
        x0_pred = model(t0, gamma, omega)
        x0_t = gradients(x0_pred, t0)

        loss_ic = torch.mean((x0_pred - x0)**2) + torch.mean((x0_t - v0)**2)

        loss = loss_ode + lambda_ic * loss_ic

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0:
            print(f"[{step:6d}] loss={loss.item():.3e}  ode={loss_ode.item():.3e}  ic={loss_ic.item():.3e}")

    return model


if __name__ == "__main__":
    _ = train_pinn()
