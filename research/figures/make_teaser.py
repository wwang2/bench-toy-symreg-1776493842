"""Generate teaser figure — scatter of noisy training points."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
})

data = np.loadtxt("research/eval/train_data.csv", delimiter=",", skiprows=1)
x, y = data[:, 0], data[:, 1]

fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax.scatter(x, y, s=30, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Toy symbolic regression — 40 noisy points, σ=0.03, x ∈ [-4, 4]")
ax.axhline(0, color="#888888", lw=0.5, alpha=0.5)
ax.axvline(0, color="#888888", lw=0.5, alpha=0.5)
fig.savefig("research/figures/teaser.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote research/figures/teaser.png")
