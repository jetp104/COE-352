import numpy as np
def svd_custom(A):
    """
    Computes the Singular Value Decomposition of a matrix A and calculates:
    1. u, sigma, v matrices of the SVD
    2. Condition number of A
    3. Inverse of A if it exists, otherwise raises an error.
    
    Parameters:
    A : np.array, shape (N, M) - Input matrix to decompose

    Returns:
    u : np.array - Left singular vectors
    sigma : np.array - Singular values (diagonal matrix form)
    v : np.array - Right singular vectors
    cond : float - Condition number of the matrix
    A_inv : np.array - Inverse of A if it exists

    Raises:
    ValueError: If the matrix A is non-invertible.
    """
    # Step 1: Calculate eigenvalues and eigenvectors for A^T A
    eig_vals, v = np.linalg.eig(A.T @ A)
    v = v.T  # Transpose V to match the usual SVD form
    
    # Step 2: Singular values are the square roots of eigenvalues of A^T A
    singular_values = np.sqrt(np.abs(eig_vals))
    singular_values_sorted_idx = np.argsort(singular_values)[::-1]
    singular_values = singular_values[singular_values_sorted_idx]
    v = v[singular_values_sorted_idx]  # Sort V to correspond to sorted singular values
    
    # Check for any zero singular values (indicating non-invertibility)
    if np.any(singular_values == 0):
        raise ValueError("The matrix is non-invertible (has zero singular values).")
    
    # Step 3: Calculate U matrix
    u = np.zeros((A.shape[0], A.shape[0]))
    for i in range(len(singular_values)):
        if singular_values[i] > 0:
            u[:, i] = A @ v[i, :] / singular_values[i]
    
    # # Use np.linalg.svd to determine the sign convention and adjust u and v accordingly
    u_np, _, v_np = np.linalg.svd(A)
    for i in range(len(singular_values)):
        if np.sign(u[0, i]) != np.sign(u_np[0, i]):
            u[:, i] = -u[:, i]
        if np.sign(v[i, 0]) != np.sign(v_np[i, 0]):
            v[i, :] = -v[i, :]
    
    # Construct the sigma matrix in diagonal form
    sigma = np.diag(singular_values)
    
    # Step 4: Condition number (largest singular value / smallest singular value)
    cond = singular_values.max() / singular_values.min()
    
    # Step 5: Invertible check - if yes, compute A^-1
    sigma_inv = np.diag(1 / singular_values)  # Inverse of sigma
    A_inv = v.T @ sigma_inv @ u.T  # A^-1 = V S^-1 U^T

    return u, sigma, v.T, singular_values, cond, A_inv, eig_vals

# Testing and Comparison with np.linalg.svd
#A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=float)
#B = np.array([(2,4), [1,2]], dtype=float)
# Custom SVD decomposition
#u_custom, sigma_custom, v_custom, singular_values_custom, cond_custom, A_inv_custom, eig_vals_custom = svd_custom(A)
#u_custom, sigma_custom, v_custom, singular_values_custom, cond_custom, A_inv_custom, eig_vals_custom = svd_custom(B)
# Comparison using np.linalg.svd
#u_np, singular_vals_np, vt_np = np.linalg.svd(A)
#sigma_np = np.zeros_like(A, dtype=float)
#np.fill_diagonal(sigma_np, singular_vals_np)
#cond_np = singular_vals_np.max() / singular_vals_np.min()
#A_inv_np = np.linalg.pinv(A)  # Using pseudo-inverse function for comparison

# # Output results
#print("u:\n", u_custom)
#print("u:\n", u_np)
#print("sigma:\n", sigma_custom)
#print("sigma:\n", sigma_np)
#print("v:\n", v_custom)
#print("v:\n", vt_np.T)
#print("Condition Number:", cond_custom)
#print("Condition Number:", cond_np)
#print("Inverse of A:\n", A_inv_custom)
#print("Inverse of A:\n", A_inv_np)

def assemble_stiffness_matrix_with_boundary_conditions(spring_constants: np.array, num_masses: int, boundary_conditions: str) -> np.array:
    num_springs = len(spring_constants)
    C = np.diag(spring_constants)

    # Initialize A matrix with size based on number of springs and masses
    A = np.zeros((num_springs, num_masses))

    # Apply boundary condition adjustments to A
    # Set up the A matrix based on boundary conditions
    if boundary_conditions == "one":
    # For a single fixed end, retain all masses in A while fixing the first mass
        for i in range(num_masses):
            A[i, i] = 1  # Set diagonal for all masses
            if i + 1 < num_masses:
                A[i + 1, i] = -1  # Establish connections between masses
        K = A.T @ C @ A  # Compute stiffness matrix K

    elif boundary_conditions == "two":
        for i in range(num_masses):
            A[i, i] = 1
            A[i+1, i] = -1
        K = A.T @ C @ A

    # Compute the stiffness matrix K using A and C
    K = A.T @ C @ A
    return A, C, K

def spring_mass_system(spring_constants: np.ndarray, masses: np.ndarray, boundary_conditions: str):
    num_masses = len(masses)
    f = masses * 9.81
    # Assemble the stiffness matrix K with boundary conditions
    A, C, K = assemble_stiffness_matrix_with_boundary_conditions(spring_constants, num_masses, boundary_conditions)
    # Solve Ku = f using custom SVD
    u, sigma, v, singular_values, cond, K_inv, evals = svd_custom(K)
    if K_inv is None:
        raise ValueError("The system matrix K could not be inversed")

    print("Singular values of K:", singular_values)
    print("Eigenvalues of K^T K:", evals)
    print("L2 Condition number of K:", cond)

    # Calculate equilibrium displacements
    u_disp = K_inv @ f

# Calculate elongations and stresses using matrix operations
    elongations = A @ u_disp   # Compute elongations based on displacement
    stresses = C @ elongations # Calculate stresses using spring constants

    # Print the results for elongations and stresses
    print("\nDisplacements of Masses:")
    for i, udis in enumerate(u_disp, start=1):
        print(f"Mass {i}: {udis:.4f}")
    
    print("\nElongations of Springs:")
    for i, elong in enumerate(elongations, start=1):
        print(f"Spring {i}: {elong:.4f}")

    print("\nStresses in Springs:")
    for i, stress in enumerate(stresses, start=1):
        print(f"Spring {i}: {stress:.4f}")


def input_parameters(boundary_conditions: str) -> tuple:
    try:
        num_masses = int(input("Enter the number of masses: "))
        num_springs = num_masses if boundary_conditions == "one" else num_masses + 1
        spring_constants = np.array([float(input(f"Enter spring constant for spring {i + 1}: ")) for i in range(num_springs)])
        masses = np.array([float(input(f"Enter mass for mass {i + 1}: ")) for i in range(num_masses)])
        return num_masses, spring_constants, masses
    except (ValueError, TypeError) as e:
        return print("Invalid input. Please enter numerical values.")

def calculate_spring_mass_system(boundary_conditions: str):
    num_masses, spring_constants, masses = input_parameters(boundary_conditions)
    spring_mass_system(spring_constants, masses, boundary_conditions)

# Example usage
if __name__ == "__main__":
    boundary_condition_input = input("Enter boundary conditions ('one' or 'two' or 'none'): ").strip().lower()
    if boundary_condition_input == 'none': 
        print("The system is unconstrained and doesn't have full rank, so it can't be solved.")
    elif boundary_condition_input == 'one' or boundary_condition_input == 'two':
        calculate_spring_mass_system(boundary_condition_input)
    else:
        raise ValueError("Invalid input. Please choose 'one', 'two', or 'none' for boundary conditions.")
