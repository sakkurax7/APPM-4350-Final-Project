import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Choose parameters that actually give a real nontrivial branch
v = 1.0
lam = 1.0
m = 1.0
g = 0.5

# Higgs potential
def V(h, phi):
    return (lam/4)*(h**2 - v**2)**2 + 0.5*(m**2 + 2*g*h)*phi**2

# Derivatives needed for curvature and determinant
def V_hh(h):
    return lam*(3*h**2 - v**2)

def det_M(h, phi):
    a = V_hh(h)
    b = 2*g*phi
    c = m**2 + 2*g*h
    return a*c - b**2

# --- Compute nontrivial branch ---
# Branch condition: m^2 + 2gh = 0
h_star = -m**2/(2*g)

# Corresponding phi* values (+ branch and – branch)
phi_sq = (2*g**2/(lam*m**2))*(v**2 - h_star**2)
phi_star_plus  = np.sqrt(phi_sq)
phi_star_minus = -np.sqrt(phi_sq)

# Points where determinant vanishes ALONG the nontrivial branch
def det_branch(phi):
    return det_M(h_star, phi)

# Zero determinant locations on the branch
phi_vals = np.linspace(-2*v, 2*v, 500)
det_vals = det_branch(phi_vals)

# Find zeros numerically
sign_changes = np.where(np.sign(det_vals[:-1]) != np.sign(det_vals[1:]))[0]
phi_deg = [phi_vals[i] for i in sign_changes]

# Domain for 3D plot
h_vals = np.linspace(-2*v, 2*v, 400)
phi_grid = np.linspace(-2*v, 2*v, 400)
H, PHI = np.meshgrid(h_vals, phi_grid)
V_vals = V(H, PHI)

# ---- Plot ----
fig = plt.figure(figsize=(14,6))

### LEFT: 3D Higgs potential ###
ax1 = fig.add_subplot(1,2,1, projection='3d')
surf = ax1.plot_surface(H, PHI, V_vals, cmap=cm.viridis, alpha=0.85)

# Mark nontrivial branch curve
ax1.plot([h_star]*len(phi_vals), phi_vals, V(h_star, phi_vals),
          color='orange', linewidth=3, label="Nontrivial branch")

# Mark degeneracy points
for ph in phi_deg:
    ax1.scatter(h_star, ph, V(h_star, ph), color='red', s=80,
                label="det M² = 0" if ph==phi_deg[0] else "")

# Mark equilibrium points (two minima)
equilibria = [(v,0),( -v,0)]
for h0,phi0 in equilibria:
    ax1.scatter(h0, phi0, V(h0,phi0), color='black', marker='D', s=60,
                label="Equilibrium" if (h0,phi0)==equilibria[0] else "")

ax1.set_title("3D Higgs Potential — Nontrivial Branch Marked")
ax1.set_xlabel("h")
ax1.set_ylabel("phi")
ax1.set_zlabel("V(h,phi)")

### RIGHT: 2D slice along nontrivial branch ###
ax2 = fig.add_subplot(1,2,2)
V_branch = V(h_star, phi_vals)
ax2.plot(phi_vals, V_branch, color='orange', linewidth=2,
         label="V(h_*, φ) along nontrivial branch")

# Degenerate points
for ph in phi_deg:
    ax2.scatter(ph, V(h_star, ph), color='red', s=80,
                label="det M² = 0" if ph==phi_deg[0] else "")

ax2.set_title("Nontrivial Branch Slice with Degeneracy Points")
ax2.set_xlabel("φ")
ax2.set_ylabel("V(h_*, φ)")
ax2.legend()

plt.tight_layout()
plt.show()
