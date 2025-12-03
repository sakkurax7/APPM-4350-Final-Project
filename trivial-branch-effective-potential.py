import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Parameters
v = 1.0
lam = 1.0
m = 0.5
g = 0.5

# Higgs potential
def V(h, phi):
    return (lam/4)*(h**2 - v**2)**2 + 0.5*(m**2 + 2*g*h)*phi**2

# Second derivative wrt h
def V_hh(h):
    return lam*(3*h**2 - v**2)

# Hessian determinant on trivial branch (phi=0)
def det_M(h):
    return V_hh(h) * (m**2 + 2*g*h)

# Domain
h_vals = np.linspace(-2*v, 2*v, 400)
phi_vals = np.linspace(-2*v, 2*v, 400)
H, PHI = np.meshgrid(h_vals, phi_vals)

# Compute potential
V_vals = V(H, PHI)

# Points where M^2 is non-invertible
h_zc_plus = v/np.sqrt(3)     # V'' = 0
h_zc_minus = -v/np.sqrt(3)

h_branch_switch = -m**2/(2*g)   # m^2 + 2 g h = 0

degenerate_points = []
for h0 in [h_zc_plus, h_zc_minus, h_branch_switch]:
    if np.abs(h0) <= 2*v:   # within plotting window
        degenerate_points.append(h0)

# Equilibria on trivial branch
equilibria = [(-v,0), (0,0), (v,0)]


# ---- Plot ----
fig = plt.figure(figsize=(14,6))

### LEFT: 3D Higgs potential ###
ax1 = fig.add_subplot(1,2,1, projection='3d')
surf = ax1.plot_surface(H, PHI, V_vals, cmap=cm.viridis, alpha=0.85)

# Mark degeneracy points on trivial branch (phi=0 line)
for h0 in degenerate_points:
    ax1.scatter(h0, 0, V(h0,0), color='red', s=80, label="Degenerate point" if h0==degenerate_points[0] else "")

# Mark equilibrium points
for h0,phi0 in equilibria:
    ax1.scatter(h0, phi0, V(h0,phi0), color='black', s=60, marker='D',
                label="Equilibrium" if (h0,phi0)==equilibria[0] else "")

ax1.set_title("3D Higgs Potential with Degeneracy Points")
ax1.set_xlabel("h")
ax1.set_ylabel(r"$\phi$")
ax1.set_zlabel(r"$V(h,\phi)$")

### RIGHT: 2D trivial branch cross-section ###
ax2 = fig.add_subplot(1,2,2)
V_line = V(h_vals, 0)
ax2.plot(h_vals, V_line, color='blue', linewidth=2, label="V(h,0)")

# Mark degeneracy points
for h0 in degenerate_points:
    ax2.scatter(h0, V(h0,0), color='red', s=80, label="det M² = 0" if h0==degenerate_points[0] else "")

# Mark equilibria
for h0,phi0 in equilibria:
    ax2.scatter(h0, V(h0,0), color='black', marker='D', s=60,
                label="Equilibrium" if h0==equilibria[0][0] else "")

ax2.set_title("Trivial Branch (φ=0) with Degeneracy Points")
ax2.set_xlabel("h")
ax2.set_ylabel("V(h,0)")
ax2.legend()

plt.tight_layout()
plt.show()
