# 1D Klein-Gordon with source: static 2x2 + side-by-side animation
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq, fftshift
from matplotlib.animation import FuncAnimation
import time

# -----------------------
# Parameters
# -----------------------
c = 1.0
L = 200.0               # domain length
N = 2048                # spatial points (power of 2 for FFT speed)
x = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]
k = 2*np.pi*fftfreq(N, d=dx)   # wave numbers (radial ordering by fftfreq)

# Initial data (zero for clarity)
f = np.zeros_like(x)
g = np.zeros_like(x)

# Source S(x,t): spatial Gaussian at center, temporal Gaussian pulse around t0
x0 = 0.0
sigma_x = 2.0
t0 = 10.0
sigma_t = 2.0
def source_xt(x_array, t):
    return np.exp(-((x_array-x0)**2)/(2*sigma_x**2)) * np.exp(-((t-t0)**2)/(2*sigma_t**2))

# Precompute spatial FFT of initial data (here zero)
fhat = fft(f)
ghat = fft(g)

# Time grid for quadrature
Tmax = 100.0
Nt = 400
ts = np.linspace(0, Tmax, Nt)
dt = ts[1] - ts[0]

# Precompute source Fourier transforms at each time sample:
# S_hat[k_idx, time_idx]
S_hat = np.empty((N, Nt), dtype=complex)
print("Computing source transforms...")
tic = time.time()
for j, tt in enumerate(ts):
    S = source_xt(x, tt)
    S_hat[:, j] = fft(S)
toc = time.time()
print(f"Computed {Nt} FFTs in {toc-tic:.2f}s")

# Evolution kernel: omega_k for given mass m
def omega_of_m(m):
    return np.sqrt(c**2 * k**2 + m**2)

# Solve (spectral Duhamel) for phi_hat(k,t) at all t:
# phi_hat(k,t_n) = fhat*cos(omega t_n) + ghat*(sin(omega t_n)/omega)
#                   + sum_{m=0}^{n} dt * sin(omega*(t_n - t_m))/omega * S_hat(k,t_m)
def compute_phi_time_series(m):
    omega = omega_of_m(m)
    # handle omega==0 safely
    small = 1e-14
    omega_safe = omega.copy()
    omega_safe[np.abs(omega_safe) < small] = 1.0  # temporary avoid div by zero; handle limit below

    # precompute cos(omega*ts) and sin(omega*ts)/omega
    cos_om_t = np.cos(np.outer(omega, ts))    # shape (N, Nt)
    sin_over_om_t = np.sin(np.outer(omega, ts)) / omega_safe[:, None]

    #sin_over_om_t = np.sin(np.outer(omega, ts)) / omega_safe  # shape (N, Nt)
    # fix entries where omega ~ 0: sin(omega t)/omega -> t
    zero_mask = np.abs(omega) < small
    if np.any(zero_mask):
        sin_over_om_t[zero_mask, :] = ts  # broadcast

    # init phi_hat array
    phi_hat = np.empty((N, Nt), dtype=complex)

    # compute homogeneous part once
    hom_part = fhat[:, None]*cos_om_t + ghat[:, None]*sin_over_om_t

    # Duhamel / time convolution: for each time n, integrate over tau <= t_n
    # simple trapezoidal rule (O(N * Nt^2) — fine for moderate Nt; reduce Nt if needed)
    print("Computing time convolution (this may take a few seconds)...")
    tic = time.time()
    # precompute S_hat * dt for speed
    Sdt = S_hat * dt
    # We'll accumulate integral incrementally to avoid O(N*Nt^2) recompute: do cumulative convolution
    # For fixed k-index, integral_n = sum_{m=0}^n dt * sin(omega_k*(t_n-t_m))/omega_k * S_hat[k,m]
    # We compute by looping over n and using vectorized operations over k
    for n in range(Nt):
        # vector of (t_n - t_m) for m=0..n
        tau = ts[n] - ts[:n+1]  # length n+1
        # kernel: sin(omega * tau)/omega  -> shape (N, n+1)
        # use broadcasting
        kernel = np.sin(np.outer(omega, tau)) / omega_safe[:, None]
        if np.any(zero_mask):
            kernel[zero_mask, :] = tau  # limit
        # integral approx: sum kernel * S_hat[:, :n+1] * dt
        conv = np.sum(kernel * S_hat[:, :n+1], axis=1) * dt
        phi_hat[:, n] = hom_part[:, n] + conv
        if (n % 50) == 0:
            pass  # progress can be printed if desired
    toc = time.time()
    print(f"Time convolution done in {toc-tic:.2f}s")
    return phi_hat

# Choose masses to compare
masses_to_run = [0.0, 1.0]

# Compute phi_hat time series for each mass (this is the heavy step)
phi_hats = {}
for m in masses_to_run:
    print(f"Computing phi_hat for m={m}")
    phi_hats[m] = compute_phi_time_series(m)

# Inverse FFT to get phi(x,t) real arrays
phis = {}
for m in masses_to_run:
    print(f"Inverting FFTs for m={m}")
    ph = np.real(ifft(phi_hats[m], axis=0))  # shape (N, Nt) with axis=0 spatial
    # reorder to x along axis 0; we want phi(x_index, time_index)
    phis[m] = ph

# -----------------------
# Static 2x2 figure: show massless top row, massive bottom row, times t= tA, tB
# -----------------------
tA = 10.0
tB = 35.0
idxA = np.argmin(np.abs(ts - tA))
idxB = np.argmin(np.abs(ts - tB))

fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
axes = axes.flatten()

plots = [(masses_to_run[0], idxA), (masses_to_run[0], idxB),
         (masses_to_run[1], idxA), (masses_to_run[1], idxB)]

# --- Handle axes dimensionality robustly ---
axes = np.atleast_1d(axes)  # ensures we can iterate over it
n_axes = len(axes)
ncols = int(np.sqrt(n_axes))  # guess grid layout if needed
nrows = n_axes // ncols if ncols > 0 else 1

for i, (ax, (m, idx)) in enumerate(zip(axes, plots)):
    phi_snap = phis[m][:, idx]
    ax.plot(x, phi_snap, color='tab:blue' if m == 0 else 'tab:red', lw=1.5)
    ax.set_title(f"m={m:.1f}, t={ts[idx]:.1f}")
    ax.grid(alpha=0.3)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-0.6, 1.2)

    # compute row/col index safely
    row, col = divmod(i, ncols)

    # label only bottom row and left column
    if row == nrows - 1:
        ax.set_xlabel("x")
    if col == 0:
        ax.set_ylabel("ϕ(x,t)")

# place main title and legend outside
fig.suptitle("1D Klein–Gordon with source: snapshots", fontsize=12, y=0.98)
plt.subplots_adjust(top=0.88, bottom=0.07, left=0.09, right=0.95, hspace=0.25)

plt.show()
