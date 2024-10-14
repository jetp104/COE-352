import numpy as np
import matplotlib.pyplot as plt

# initialzie variables 
alpha = 1
t_max = 5
y0 = 1

# exact solution we dervied
def exact_solution(t):
    return np.exp(-alpha * t)

#Forward Euler function 
def forward_euler(alpha, delta_t, t_max, y0):
    N_steps = int(t_max / delta_t)
    t_values = np.linspace(0, t_max, N_steps + 1)
    y_values = np.zeros(N_steps + 1)
    
    y_values[0] = y0
    
    for n in range(N_steps):
        y_values[n + 1] = y_values[n] * (1 - alpha * delta_t)
    
    return t_values, y_values

# Time steps that we test 
delta_t_values = [0.1,0.5,1.0,2.5]

# Plotting of the time steps using forward Euler method 
plt.figure(figsize=(10, 6))

for delta_t in delta_t_values:
    t_values, y_numerical = forward_euler(alpha, delta_t, t_max, y0)
    y_exact = exact_solution(t_values)
    
    plt.plot(t_values, y_numerical, label=f'Numerical (∆t = {delta_t})')
    
    if delta_t == delta_t_values[0]:
        plt.plot(t_values, y_exact, 'k--', label='Exact Solution')

plt.title('Forward Euler: Numerical vs Exact Solution')
plt.xlabel('Time t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.show()

# Backwards Euler method 
def backward_euler(alpha, delta_t, t_max, y0):
    N_steps = int(t_max / delta_t)
    t_values = np.linspace(0, t_max, N_steps + 1)
    y_values = np.zeros(N_steps + 1)
    
    y_values[0] = y0
    
    for n in range(N_steps):
        y_values[n + 1] = y_values[n] / (1 + alpha * delta_t)
    
    return t_values, y_values

# Time steps used 
delta_t_values = [ 0.1, 0.5, 1.0, 2.5]

# Plotting the backwards Euler method 
plt.figure(figsize=(10, 6))

for delta_t in delta_t_values:
    t_values, y_numerical = backward_euler(alpha, delta_t, t_max, y0)
    y_exact = exact_solution(t_values)
    
    plt.plot(t_values, y_numerical, label=f'Numerical (∆t = {delta_t})')
    
    if delta_t == delta_t_values[0]:
        plt.plot(t_values, y_exact, 'k--', label='Exact Solution')

plt.title('Backward Euler: Numerical vs Exact Solution')
plt.xlabel('Time t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.show()

# Trapezoid method 
def trapezoid_method(alpha, delta_t, t_max, y0):
    N_steps = int(t_max / delta_t)
    t_values = np.linspace(0, t_max, N_steps + 1)
    y_values = np.zeros(N_steps + 1)
    
    y_values[0] = y0
  
    for n in range(N_steps):
        y_values[n+1] = ((1 - alpha * delta_t / 2) / (1 + alpha * delta_t / 2)) * y_values[n]
    
    return t_values, y_values
# Time steps used 
delta_t_values = [0.1, 0.5, 1.0, 2.5]

# Plotting the trapezoid method 
plt.figure(figsize=(10, 6))

for delta_t in delta_t_values:
    t_values, y_numerical = trapezoid_method(alpha, delta_t, t_max, y0)
    y_exact = exact_solution(t_values)
    
    plt.plot(t_values, y_numerical, label=f'Numerical (∆t = {delta_t})')
    
    if delta_t == delta_t_values[0]:
        plt.plot(t_values, y_exact, 'k--', label='Exact Solution')

plt.title('Trapezoid Method: Numerical vs Exact Solution')
plt.xlabel('Time t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.show()