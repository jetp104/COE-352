import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x)
def f(x):
    return 4 + 8*x**2 - x**4

# Generate x values
x = np.linspace(-3, 3, 400)
y = f(x)

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r"$f(x) = 4 + 8x^2 - x^4$", color='orange')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.title("Graph of $f(x) = 4 + 8x^2 - x^4$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()

# Check if the function is even or odd
x_values = np.linspace(-10, 10, 1000)
f_x = f(x_values)
f_neg_x = f(-x_values)

if np.allclose(f_x, f_neg_x):
    print("The function is even.")
elif np.allclose(f_x, -f_neg_x):
    print("The function is odd.")
else:
    print("The function is neither even nor odd.")

def f_prime(x):
    return 16*x - 4*x**3

# Implement Newton's Method
def newtons_method(x0, num_iterations):
    x = x0
    for i in range(num_iterations):
        print(f"Iteration {i+1}: x = {x:.6f}, f(x) = {f(x):.6f}")
        x = x - f(x) / f_prime(x)
    return x

# Initial guess and number of iterations
x0 = 3
num_iterations = 2

# Perform Newton's Method
x_approx = newtons_method(x0, num_iterations)

print(f"\nApproximate x-intercept after {num_iterations} iterations: x = {x_approx:.6f}")

def f(x):
    return x**3 - 3*x - 3

# Generate x values
x = np.linspace(-3, 3, 400)
y = f(x)

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r"$f(x) = x^3 - 3x - 3$", color='blue')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.title("Graph of $f(x) = x^3 - 3x - 3$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()

# Check if the function is even or odd
x_values = np.linspace(-10, 10, 1000)
f_x = f(x_values)
f_neg_x = f(-x_values)

if np.allclose(f_x, f_neg_x):
    print("The function is even.")
elif np.allclose(f_x, -f_neg_x):
    print("The function is odd.")
else:
    print("The function is neither even nor odd.")

def f_prime(x):
    return 3*x**2 - 3

# Implement Newton's Method
def newtons_method(x0, num_iterations):
    x = x0
    for i in range(num_iterations):
        print(f"Iteration {i+1}: x = {x:.6f}, f(x) = {f(x):.6f}")
        x = x - f(x) / f_prime(x)
    return x

# Initial guess and number of iterations
x0 = 2
num_iterations = 2

# Perform Newton's Method
x_approx = newtons_method(x0, num_iterations)

print(f"\nApproximate x-intercept after {num_iterations} iterations: x = {x_approx:.6f}")

def bisection_method(f, a, b, tol):
    if f(a) * f(b) >= 0:
        print("Bisection method fails. f(a) and f(b) must have opposite signs.")
        return None

    while (b - a) / 2.0 > tol:
        midpoint = (a + b) / 2.0
        if f(midpoint) == 0:
            return midpoint  # Found exact root
        elif f(a) * f(midpoint) < 0:
            b = midpoint  # Root is in the left half
        else:
            a = midpoint  # Root is in the right half

    return (a + b) / 2.0  # Return midpoint as the root

# Define the function for bisection
def f_bisection(x):
    return x**2 - 3

# Initial interval and tolerance
a = 1
b = 2
tolerance = 1e-5

# Perform Bisection Method
root = bisection_method(f_bisection, a, b, tolerance)

print(f"Root found by Bisection Method: x = {root:.6f}")
