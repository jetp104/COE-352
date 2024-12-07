import numpy as np
import matplotlib.pyplot as plt

# Define the right-hand side function f(x, t)
f = lambda x, t: (np.pi**2 - 1) * np.exp(-t) * np.sin(np.pi * x)

# Exact solution
u_exact = lambda x, t: np.exp(-t) * np.sin(np.pi * x)

# Function to form the global stiffness and mass matrices
def form_matrices(N, xl, xr):
    # Uniform grid and connectivity
    Ne = N - 1
    h = (xr - xl) / (N - 1)
    x = np.linspace(xl, xr, N)
    iee = np.zeros((Ne, 2), dtype=int)
    for i in range(Ne):
        iee[i, 0] = i
        iee[i, 1] = i + 1

    # Basis functions and their derivatives
    p = lambda l, g: (1 + (-1)**l * g) / 2
    dp = lambda l, g: 0.5 * (-1)**l

    # Gaussian quadrature weights and points (2nd order)
    C = [1, 1]
    g = [-1 / np.sqrt(3), 1 / np.sqrt(3)]

    # Initialize global matrices
    K = np.zeros((N, N))  # Stiffness matrix
    M = np.zeros((N, N))  # Mass matrix

    # Local matrices
    klocal = np.zeros((2, 2))
    mlocal = np.zeros((2, 2))

    # Assemble global matrices
    for k in range(Ne):
        for l in range(2):
            for m in range(2):
                klocal[l, m] = sum(dp(l, g[j]) * dp(m, g[j]) * C[j] for j in range(len(g))) * 2 / h
                mlocal[l, m] = sum(p(l, g[j]) * p(m, g[j]) * C[j] for j in range(len(g))) * h / 2

        for l in range(2):
            global_node1 = iee[k, l]
            for m in range(2):
                global_node2 = iee[k, m]
                K[global_node1, global_node2] += klocal[l, m]
                M[global_node1, global_node2] += mlocal[l, m]

    # Apply boundary conditions to the matrices
    for i in range(N):
        if i == 0 or i == N - 1:
            for j in range(N):
                if i != j:
                    K[j, i] = 0
                    K[i, j] = 0
                    M[j, i] = 0
                    M[i, j] = 0
            K[i, i] = 1
            M[i, i] = 1

    return x, K, M, iee, h, g, C

# Function to solve the 1D heat equation using Forward Euler
def forward_euler(N, T0, Tf, dt, x, K, M, iee, h, g, C):
    # Compute the inverse of the mass matrix
    M_inv = np.linalg.inv(M)

    # Initial condition
    u = np.sin(np.pi * x)

    # Time stepping
    ctime = T0
    nt = int((Tf - T0) / dt)

    for n in range(nt):
        ctime = T0 + n * dt

        # Right-hand side vector
        F = np.zeros(N)
        for k in range(N - 1):
            flocal = np.zeros(2)
            for l in range(2):
                flocal[l] = sum(
                    f((g[j] + 1) * h / 2 + x[k], ctime) * (1 + (-1)**l * g[j]) / 2 * C[j] for j in range(len(g))
                ) * h / 2
            for l in range(2):
                global_node = iee[k, l]
                F[global_node] += flocal[l]

        # Apply boundary conditions to F
        F[0] = 0
        F[-1] = 0

        # Update the solution using Forward Euler
        u = u - dt * (M_inv @ K @ u) + dt * (M_inv @ F)
    print(F)
    return u


# Function to solve the 1D heat equation using Forward Euler
def backward_euler(N, T0, Tf, dt, x, K, M, iee, h, g, C):
    # Compute the inverse of the mass matrix
    M_inv = np.linalg.inv(M)

    # Initial condition
    u = np.sin(np.pi * x)
    B = (1/dt)*M+K 
    # Time stepping
    ctime = T0
    nt = int((Tf - T0) / dt)

    for n in range(nt):
        ctime = T0 + n * dt

        # Right-hand side vector
        F = np.zeros(N)
        for k in range(N - 1):
            flocal = np.zeros(2)
            for l in range(2):
                flocal[l] = sum(
                    f((g[j] + 1) * h / 2 + x[k], ctime) * (1 + (-1)**l * g[j]) / 2 * C[j] for j in range(len(g))
                ) * h / 2
            for l in range(2):
                global_node = iee[k, l]
                F[global_node] += flocal[l]

        # Apply boundary conditions to F
        F[0] = 0
        F[-1] = 0

        # Update the solution using Forward Euler
        u = np.linalg.inv(B).dot((1 / dt) * M.dot(u) + F)  
 
    return u

# Parameters
N = int(input("Enter the number of nodes: "))  # Number of nodes
xl = 0  # Left boundary
xr = 1  # Right boundary
T0 = 0  # Initial time
Tf = 1  # Final time
dt = 1/int(input("Enter the number of steps: ")) # Time step
#dt = 1/539
# Form the global matrices
x, K, M, iee, h, g, C = form_matrices(N, xl, xr)

# Solve the heat equation using Forward Euler
Choice = input("Enter FE or BE for method: " )
if Choice == 'FE': 
    u_numerical = forward_euler(N, T0, Tf, dt, x, K, M, iee, h, g, C)
    # Plot the solutions
    # Define a fixed grid for the exact solution
    x_fixed = np.linspace(xl, xr, 100)  # Use a fixed grid with high resolution
    u_exact_solution = u_exact(x_fixed, Tf)
    plt.plot(x, u_numerical, label="Numerical Solution")
    plt.plot(x_fixed, u_exact_solution, label="Exact Solution", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.title("1D Heat Equation Solution (Forward Euler)")
    plt.legend()
    plt.grid()
    plt.show()
elif Choice == 'BE': 
    u_back = backward_euler(N, T0, Tf, dt, x, K, M, iee, h, g, C)
    # Define a fixed grid for the exact solution
    x_fixed = np.linspace(xl, xr, 100)  # Use a fixed grid with high resolution
    u_exact_solution = u_exact(x_fixed, Tf)

    plt.plot(x, u_back, label="Numerical Solution")
    plt.plot(x_fixed, u_exact_solution, label="Exact Solution", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.title("1D Heat Equation Solution (Backward Euler)")
    plt.legend()
    plt.grid()
    plt.show()

print("Instability for FE occurs when the time step is delta t = 1/539.")
print("With less nodes the accaurcy of the model decreases")
print("When the time-step is equal to or greater than the spatial size the model's accuarcy once again decreases. This occurs because the backward Euler method, although stable, is only first-order accurate in time. As the time-step (delta 𝑡) increases, the solution becomes less sensitive to rapid changes, resulting in significant temporal discretization errors and reduced accuracy due to the method's low temporal order.")
