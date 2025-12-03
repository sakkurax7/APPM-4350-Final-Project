import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq
from matplotlib.animation import FuncAnimation

# ------------------------
# Simulation parameters
# ------------------------
N = 512             # grid size (NxN)
L = 20.0            # domain half-width ~ we use [-L/2, L/2]
c = 1.0
a = 2.0             # initial Gaussian width
masses = [0.0, 1.0] # m=0 (massless) and m=1 (massive)

# spatial grid
x = np.linspace(-L/2, L/2, N, endpoint=False)
y = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# initial data: centered Gaussian, zero initial velocity
f = np.exp(-(X**2 + Y**2)/a**2)
g = np.zeros_like(f)

# Fourier domain set up (2D)
kx = 2*np.pi * fftfreq(N, d=L/N)
ky = 2*np.pi * fftfreq(N, d=L/N)
KX, KY = np.meshgrid(kx, ky)
k2 = KX**2 + KY**2

# FFT of initial conditions (compute once)
fhat = fft2(f)
ghat = fft2(g)

# evolution function, returns real-space phi(x,y,t) for a given mass m
def phi_t(t, m):
    omega_k = np.sqrt(c**2 * k2 + m**2)
    # safe sin(omega*t)/omega, avoid division by zero (use limit t at omega~0)
    sin_over_omega = np.zeros_like(omega_k)
    mask = omega_k > 1e-12
    sin_over_omega[mask] = np.sin(omega_k[mask]*t)/omega_k[mask]
    sin_over_omega[~mask] = t  # limit as omega->0
    phat = fhat * np.cos(omega_k * t) + ghat * sin_over_omega
    return np.real(ifft2(phat))

# ------------------------
# Static 2x2 figure
# Layout: rows = [massless, massive], cols = [t0, t1]
# ------------------------
times = [0.0, 5.0]  # two times; we'll make 2x2 (massless row, massive row)
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), sharex=True, sharey=True)
cmap = 'RdBu_r'
vmax = 1.0
vmin = -1.0

for row, m in enumerate(masses):
    for col, t in enumerate(times):
        ax = axes[row, col]
        phi = phi_t(t, m)
        im = ax.imshow(phi, extent=[-L/2, L/2, -L/2, L/2],
                       origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"m = {m:.1f}, t = {t:.1f}", fontsize=10, pad=4)
        ax.set_xticks([-10, 0, 10])
        ax.set_yticks([-10, 0, 10])
        if row == 1:
            ax.set_xlabel("x")
        if col == 0:
            ax.set_ylabel("y")

# global colorbar on the right
cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.64])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, label=r'$\phi(x,y,t)$')

# legend / title placement above figure
labels = ["Initial Gaussian, g=0"]
# create a small legend box above
fig.text(0.5, 0.97, "2D Klein–Gordon evolution — massless (top) vs massive (bottom)",
         ha='center', va='top', fontsize=12)

plt.subplots_adjust(left=0.07, right=0.9, top=0.9, bottom=0.08, hspace=0.25, wspace=0.18)
plt.show()
