from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
def plot_prediction(t: np.ndarray, x_true: np.ndarray, x_pred: np.ndarray, title: str, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5,4.5))
    plt.plot(t, x_true, label="analytic")
    plt.plot(t, x_pred, "--", label="PINN")
    plt.xlabel("t"); plt.ylabel("x(t)")
    plt.title(title); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_residual(t: np.ndarray, res: np.ndarray, title: str, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5,4.5))
    plt.plot(t, res)
    plt.yscale("symlog")
    plt.xlabel("t"); plt.ylabel("residual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
def plot_curves(t: np.ndarray, curves: List[Tuple[str, np.ndarray]], title: str, outfile: str | None = None) -> None:
    plt.figure(figsize=(7.5, 4.5))
    for label, y in curves:
        plt.plot(t, y, label=label)
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, dpi=160)
    else:
        plt.show()
    plt.close()