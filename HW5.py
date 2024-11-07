import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x) = sin(pi * x)
def f(x):
    return np.sin(np.pi * x)

# Define nodes and function values
nodes = np.linspace(0, 1, 5)  # x_0, x_1, x_2, x_3, x_4
f_values = f(nodes)

# Define the piecewise linear basis function phi_i for node i
def phi(i, x, nodes):
    if i == 0:
        return (nodes[i+1] - x) / (nodes[i+1] - nodes[i]) if nodes[i] <= x <= nodes[i+1] else 0
    elif i == len(nodes) - 1:
        return (x - nodes[i-1]) / (nodes[i] - nodes[i-1]) if nodes[i-1] <= x <= nodes[i] else 0
    else:
        if nodes[i-1] <= x <= nodes[i]:
            return (x - nodes[i-1]) / (nodes[i] - nodes[i-1])
        elif nodes[i] <= x <= nodes[i+1]:
            return (nodes[i+1] - x) / (nodes[i+1] - nodes[i])
    return 0

# Construct the finite element interpolant f_h(x) using the basis functions
def f_h(x):
    result = 0.0
    for i in range(len(nodes)):
        result += f_values[i] * phi(i, x, nodes)
    return result

# Vectorize f_h to apply it to arrays
f_h_vec = np.vectorize(f_h)

# Define a dense set of x points for plotting
x_dense = np.linspace(0, 1, 100)

# Plot f(x) and f_h(x)
plt.plot(x_dense, f(x_dense), label=r"$f(x) = \sin(\pi x)$", color="blue")
plt.plot(x_dense, f_h_vec(x_dense), label=r"$f_h(x)$ (Finite Element Interpolant)", linestyle="--", color="red")
plt.scatter(nodes, f_values, color="black", zorder=5, label="Mesh Nodes")

# Formatting the plot
plt.xlabel("x")
plt.ylabel("Function Value")
plt.legend()
plt.title("Finite Element Interpolant of $f(x) = \sin(\pi x)$ Using Piecewise Linear Basis Functions")
plt.grid(True)
plt.show()

# Define the exact solution function
def exact_solution(x):
    return (-x**3 / 6) + (x / 6)

# Define the approximate solution y_h(x) from Galerkin approximation
def approximate_solution(x):
    # Constants derived for the coefficients
    c1 = 2 / np.pi**3
    c2 = -1 / (4*np.pi**3) 
    c3 = 2 / (27 * np.pi**3)
    # Basis functions
    phi1 = np.sin(np.pi * x)
    phi2 = np.sin(2 * np.pi * x)
    phi3 = np.sin(3 * np.pi * x)
    # Approximate solution
    return c1 * phi1 + c2 * phi2 + c3 * phi3

# Generate x values for plotting
x_values = np.linspace(0, 1, 100)
y_exact = exact_solution(x_values)
y_approx = approximate_solution(x_values)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_exact, label="Exact Solution", color="blue", linewidth=2)
plt.plot(x_values, y_approx, label="Galerkin Approximation", color="orange", linestyle="--", linewidth=2)
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Exact Solution vs. Galerkin Approximation")
plt.legend()
plt.grid(True)
plt.show()

# Define the exact solution (analytical solution)
def exact_solution(x):
    return -x**3 / 6 + x / 6

# Define the piecewise basis functions on intervals with h = 0.25
def phi_1(x):
    if 0 <= x < 0.25:
        return 4 * x
    elif 0.25 <= x < 0.5:
        return 2 - 4 * x
    else:
        return 0

def phi_2(x):
    if 0.25 <= x < 0.5:
        return 4 * x - 1
    elif 0.5 <= x < 0.75:
        return 3 - 4 * x
    else:
        return 0

def phi_3(x):
    if 0.5 <= x < 0.75:
        return 4 * x - 2
    elif 0.75 <= x <= 1:
        return 4 - 4 * x
    else:
        return 0

# Define the approximate solution y_h(x) using the coefficients from your notes
def approximate_solution(x):
    return (5 / 128) * phi_1(x) + 1/16 * phi_2(x) + (7 / 128) * phi_3(x)

# Generate x values for plotting
x_values = np.linspace(0, 1, 100)

# Calculate exact and approximate solutions
y_exact = [exact_solution(x) for x in x_values]
y_approx = [approximate_solution(x) for x in x_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_exact, label="Exact Solution", color="blue")
plt.plot(x_values, y_approx, label="Galerkin Approximate Solution", color="orange", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Exact Solution vs Galerkin Approximate Solution")
plt.grid(True)
plt.show()
