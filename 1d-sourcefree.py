import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 1.0
A = 1.0
k = 2 * np.pi / 5
x = np.linspace(-20, 20, 1000)

# Masses and colors
masses = [0, 1]
labels = ["Massless", "Massive"]
colors = ["tab:blue", "tab:red"]

# Time slices
times = [0, 2, 4, 6]

# Create 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5), sharex=True, sharey=True)
axes = axes.flatten()

# Plot waves
for i, t in enumerate(times):
    for m, label, color in zip(masses, labels, colors):
        omega = np.sqrt(c**2 * k**2 + (m * c**2)**2)
        phi = A * np.cos(k * x - omega * t)
        axes[i].plot(x, phi, color=color, lw=1.8, label=label if i == 0 else "")
    axes[i].set_title(f"t = {t}", fontsize=10, pad=4)
    axes[i].grid(True, alpha=0.4)
    axes[i].set_xlim(-20, 20)
    axes[i].set_ylim(-1.1, 1.1)
    if i in [2, 3]:
        axes[i].set_xlabel("x")
    if i in [0, 2]:
        axes[i].set_ylabel("ϕ(x, t)")

# --- Handle title and legend properly ---
# Add legend above figure in a separate area
handles, lbls = axes[0].get_legend_handles_labels()

fig.suptitle("1D Klein–Gordon Evolution: Massless vs Massive Waves", fontsize=11, y=.99)

# Adjust layout — do NOT use tight_layout
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.3, wspace=0.25)

plt.show()