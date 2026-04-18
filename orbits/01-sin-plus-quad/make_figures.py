"""Generate results.png and narrative.png for orbit 01-sin-plus-quad."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "eval"))
sys.path.insert(0, os.path.dirname(__file__))

from generate_data import generate_test_data, generate_train_data  # noqa: E402
from solution import f  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "medium",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 10.0,
        "axes.labelpad": 6.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.borderpad": 0.3,
        "legend.handletextpad": 0.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.constrained_layout.use": True,
    }
)

# Muted palette
C_TRAIN = "#888888"     # neutral gray for noisy observations
C_TEST = "#4C72B0"      # blue for ground truth
C_FIT = "#DD8452"       # warm orange for our closed-form fit
C_EVEN = "#55A868"      # green for even part
C_ODD = "#8172B3"       # purple for odd part

# ------------------------------------------------------------------ data
x_train, y_train = generate_train_data()
x_test, y_test = generate_test_data()
y_pred_test = f(x_test)
mse = float(np.mean((y_pred_test - y_test) ** 2))

# ------------------------------------------------------------------ results.png
fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5), constrained_layout=True)
ax.scatter(
    x_train,
    y_train,
    s=28,
    color=C_TRAIN,
    alpha=0.75,
    edgecolor="white",
    linewidth=0.5,
    label="train (noisy, n=40)",
    zorder=2,
)
ax.plot(
    x_test,
    y_test,
    color=C_TEST,
    linewidth=3.2,
    alpha=0.55,
    label="test target (clean)",
    zorder=3,
)
ax.plot(
    x_test,
    y_pred_test,
    color=C_FIT,
    linewidth=1.6,
    linestyle="--",
    dashes=(4, 3),
    label=r"$f(x)=\sin(x)+0.1\,x^{2}$  (overlays target exactly)",
    zorder=4,
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(-4.1, 4.1)
ax.set_title(f"Closed-form fit on [-4, 4]   |   test MSE = {mse:.3e}")
ax.legend(loc="upper center")
fig.savefig(
    os.path.join(FIGDIR, "results.png"),
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)

# ------------------------------------------------------------------ narrative.png
# (a) training residuals
resid = y_train - f(x_train)
resid_std = float(np.std(resid))

# (b) even/odd decomposition of TRAINING data
# pair up x[i] with x[-1-i]. Training grid is symmetric: x = linspace(-4,4,40),
# so x[-1-i] == -x[i]. Use only the non-negative half to avoid duplication.
n = len(x_train)
half = n // 2
x_pos = x_train[half:]              # x >= 0 (actually x > 0 because n is even)
y_pos = y_train[half:]
y_neg = y_train[:half][::-1]        # y at -x_pos, reversed to align
# For even n, x_train[half] > 0 and x_train[half-1] = -x_train[half]
even_emp = 0.5 * (y_pos + y_neg)
odd_emp = 0.5 * (y_pos - y_neg)
even_theory = 0.1 * x_pos**2
odd_theory = np.sin(x_pos)

fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), constrained_layout=True)

axa = axes[0]
axa.axhline(0.0, color=C_TEST, linewidth=1.2, linestyle="--", alpha=0.7)
axa.scatter(
    x_train,
    resid,
    s=36,
    color=C_FIT,
    alpha=0.85,
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
)
axa.fill_between(
    [-4.2, 4.2],
    [-0.03, -0.03],
    [0.03, 0.03],
    color=C_TEST,
    alpha=0.08,
    label=r"$\pm\sigma_{\mathrm{noise}} = 0.03$",
)
axa.set_xlabel("x")
axa.set_ylabel(r"$y_{\mathrm{train}} - f(x)$")
axa.set_xlim(-4.2, 4.2)
axa.set_title(f"(a) Training residuals  (std = {resid_std:.3f})")
axa.legend(loc="upper right")
axa.text(-0.12, 1.05, "(a)", transform=axa.transAxes, fontsize=14, fontweight="bold")

axb = axes[1]
# Even part
axb.scatter(
    x_pos,
    even_emp,
    s=36,
    color=C_EVEN,
    alpha=0.85,
    edgecolor="white",
    linewidth=0.5,
    label=r"even: $(y(x)+y(-x))/2$",
    zorder=3,
)
axb.plot(
    x_pos,
    even_theory,
    color=C_EVEN,
    linewidth=2.0,
    linestyle="--",
    label=r"$0.1\,x^{2}$",
    zorder=2,
)
# Odd part
axb.scatter(
    x_pos,
    odd_emp,
    s=36,
    color=C_ODD,
    alpha=0.85,
    edgecolor="white",
    linewidth=0.5,
    marker="s",
    label=r"odd: $(y(x)-y(-x))/2$",
    zorder=3,
)
axb.plot(
    x_pos,
    odd_theory,
    color=C_ODD,
    linewidth=2.0,
    linestyle="--",
    label=r"$\sin(x)$",
    zorder=2,
)
axb.axhline(0.0, color="#bbbbbb", linewidth=0.8, zorder=1)
axb.set_xlabel("x  (non-negative half)")
axb.set_ylabel("y component")
axb.set_title("(b) Parity decomposition of training data")
axb.legend(loc="upper left")
axb.text(-0.12, 1.05, "(b)", transform=axb.transAxes, fontsize=14, fontweight="bold")

fig.suptitle(
    "Why the closed form is right: residuals look like pure noise; parity matches",
    fontsize=13,
    y=1.04,
)
fig.savefig(
    os.path.join(FIGDIR, "narrative.png"),
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)

print(f"Wrote figures to {FIGDIR}")
print(f"MSE = {mse:.10e}")
print(f"Residual std = {resid_std:.6f}  (expected ~0.030)")
