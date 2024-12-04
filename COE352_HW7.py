import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import dblquad

# Define the original function f(x, y)
def f_original(x, y):
    return (1 / 4) * (1 - x - y + x**2 * y)

# Define the interpolated (simplified) function f(x, y)
def f_interpolated(x, y):
    return (1 - x) / 4

# Create meshgrid for plotting
x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)

# Evaluate the functions on the meshgrid
Z_original = f_original(X, Y)
Z_interpolated = f_interpolated(X, Y)

# Plot the original function and the interpolated function in 3D
fig = plt.figure(figsize=(12, 6))

# Plot original function
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z_original, cmap='viridis')
ax1.set_title('Original Function')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')

# Plot interpolated function
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z_interpolated, cmap='viridis')
ax2.set_title('Interpolated Function')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x, y)')

plt.show()

# Perform numerical integration for both functions using scipy's dblquad
# For f_original: integrate over y first, then x
integral_original, error_original = dblquad(f_original, -1, 1, lambda x: -1, lambda x: 1)

# For f_interpolated: integrate over y first, then x
integral_interpolated, error_interpolated = dblquad(f_interpolated, -1, 1, lambda x: -1, lambda x: 1)

# Output the result of the integration
print("Numerical integration result for the original function:", integral_original)
print("Numerical integration result for the interpolated function:", integral_interpolated)