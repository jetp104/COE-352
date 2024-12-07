# 1D Galerkin Solver

## Description
This project is a 1D Finite Element Method (FEA) solver using Galerkin's method with 1D Lagrange basis functions and 2nd order Gaussian Quadrature for numerical integration, also implementing both Forward and Backward Euler approximations.

## Heat Transfer Problem
Consider the heat transfer problem defined by the equation:

$$
u_t - u_{xx} = f(x, t), \quad (x, t) \in (0,1) \times (0,1)
$$

with initial and Dirichlet boundary conditions:

$$
u(x, 0) = \sin(\pi x)
$$

$$
u(0, t) = u(1, t) = 0
$$

and the function $f(x,t)$ defined as:

$$
f(x, t) = (\pi^2 - 1)e^{-t} \sin(\pi x)
$$

The analytic solution to this problem is:

$$
u(x, t) = e^{-t} \sin(\pi x)
$$

This problem is solved using the Galerkin method within the context of the 1D FEA solver to approximate the temperature distribution over time in a 1D domain.

## Weak Form Derivation
We are asked to derive the weak form of the equation by hand. The weak form of the equation is derived as follows:
![Weak Form Derivation](derivation.PNG)

## Forward Euler Implementation
The second part of the problem was the implementation of **Forward Euler time derivative discretization** with a time-step of **Δt = 1/551**. The time-stepping method was applied to solve the heat equation, and the results were plotted against the exact solution.

After running the model and plotting the results, we obtained the following:
![image](https://github.com/user-attachments/assets/af431cf0-f44a-4ec5-a23b-99ebfe695c6b)

As we can see the approximation of the Forward Euler method was very close to the exact solution given 11 nodes and a time-step of **Δt = 1/551**

Interestingly instability for this method in the case of this problem occured at around **Δt = 1/539** and looked like this: 

![image](https://github.com/user-attachments/assets/fd17a352-b59b-43f9-9a89-f0d538bf93fa)

Another important thing to note is that the model's accuracy depends on the number of nodes used in the finite element discretization. When fewer than 11 nodes are used, the solution becomes significantly less accurate. This is because a lower number of nodes leads to a coarser spatial discretization, which cannot capture the finer details of the solution, especially for problems involving steep gradients or high-frequency components. This can be seen below where we drop the node number from 11 to 5: 
![image](https://github.com/user-attachments/assets/adce0b4d-cd78-4a94-9279-de1d6ec65d24)

## Backward Euler Implementation
The final part of the problem was the implementation of **Implicit Backward Euler time derivative discretization** with a time-step of **Δt = 1/551**. The time-stepping method was applied to solve the heat equation, and the results were plotted against the exact solution.

After running the model for this problem and plotting the results we found the below graph: 
![image](https://github.com/user-attachments/assets/276d74e2-3b1c-47fa-9d25-c47a868e3e3d)

This being very similar to the exact solution and looking identical to the Forward Euler Solution showing that our model is very accuarte for the initial paramters of 11 nodes and a Δt = 1/551 for both cases. 

Finally, increasing the time-step past the spatial step size resulted in the following graph: 

![image](https://github.com/user-attachments/assets/00b119c5-327c-44c4-8717-6f9c909fc6bf)

We can now see the graph is overfit as a result of the large increase in the time-step. This is because when the time step (Δ𝑡) is large, particularly when it is comparable to or larger than the spatial step size (Δx), the accuracy of the Backward Euler method degrades. While the method is unconditionally stable, it is only first-order accurate in time, meaning that large time steps introduce substantial temporal discretization errors. This causes the solution to become less responsive to rapid changes, as the method effectively smooths out transient features. Additionally, in diffusion-dominated problems like the heat equation, larger time steps exacerbate this smoothing effect, leading to an overly diffused solution. The numerical solution may appear to "overfit" the coarse spatial grid, aligning closely with the discrete points but failing to accurately represent the underlying continuous solution. This is because the implicit nature of the Backward Euler method prioritizes stability over accuracy, particularly when the time step is too large relative to the spatial resolution. Consequently, the solution loses its ability to capture fine details of the dynamics, underscoring the importance of selecting a time step that is small enough to balance both temporal and spatial accuracy while avoiding excessive diffusion.


