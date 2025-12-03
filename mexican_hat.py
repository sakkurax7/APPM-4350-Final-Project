import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
v = -3.0       # Vacuum expectation value
lambda_ = 1.0 # Coupling constant
mu2 = 2.5     # Parameter for central spike (adds quadratic term)

# Create a grid of points (x, y)
x = np.linspace(-3.25, 3.25, 400)
y = np.linspace(-3.25, 3.25, 400)
X, Y = np.meshgrid(x, y)

# Calculate r (distance from the center)
R = np.sqrt(X**2 + Y**2)

# Mexican hat potential with a pronounced spike at the center
Z = lambda_ * (R**2 - v**2)**2 + mu2 * R**2

# Create a radial profile (2D projection of the potential)
r_values = np.linspace(-5, 5, 400)
V_values = lambda_ * (r_values**2 - v**2)**2 + mu2 * r_values**2

# Set up the figure with 2 subplots (3D plot + 2D projection)
fig = plt.figure(figsize=(14, 6))

# 3D plot of the Mexican Hat potential
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='RdBu_r', edgecolor='none', alpha=1.0)  # Make the surface opaque

# Labels for the 3D plot
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('V(r)')
ax1.set_title('Mexican Hat Potential')

# 2D projection (radial profile) plot
ax2 = fig.add_subplot(122)
ax2.plot(r_values, V_values, color='blue')
#ax2.fill_between(r_values, V_values, color='blue', alpha=0.5)
ax2.set_xlabel('r')
ax2.set_ylabel('V(r)')
ax2.set_title('2D Projection (Radial Profile)')

# Show plot
plt.tight_layout()
plt.show()
