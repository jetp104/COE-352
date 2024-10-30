For this code the usage of the SVD function was tested using the Matrix shown below. 

### Matrix
$$
\begin{bmatrix}
1 & 2 & 3 \\
0 & 1 & 4 \\
5 & 6 & 0
\end{bmatrix}
$$

which comparing my created SVD function to the SVD blackbox function give the results below: 

# Comparison of Matrices

Below are the matrices generated from the custom function and the black box function:

## Custom Function Output VS Black Box function

**Matrix \( U \):**

$$
U = \begin{bmatrix}
-0.33780332 & -0.51318841 & -0.78900353 \\
-0.19885965 & -0.78044262 & 0.59275978 \\
-0.91996943 & 0.35713718 & 0.16158364 
\end{bmatrix}
$$

$$
U_{\text{black box}} = \begin{bmatrix}
-0.33780332 & -0.51318841 & -0.78900353 \\
-0.19885965 & -0.78044262 & 0.59275978 \\
-0.91996943 & 0.35713718 & 0.16158364 
\end{bmatrix}
$$

**Matrix \( Σ \):**

$$
Σ = \begin{bmatrix}
8.27883923 & 0 & 0 \\
0 & 4.84357297 & 0 \\
0 & 0 & 0.02493818 
\end{bmatrix}
$$

$$
Σ_{\text{black box}} = \begin{bmatrix}
8.27883923 & 0 & 0 \\
0 & 4.84357297 & 0 \\
0 & 0 & 0.02493818 
\end{bmatrix}
$$

**Matrix \( $V^{T}$ \):**

$$
 V^{T} = \begin{bmatrix}
-0.59641821 & 0.26271876 & 0.75846171 \\
-0.77236466 & 0.06937103 & -0.63137983 \\
-0.2184906 & -0.96237545 & 0.16154054 
\end{bmatrix}
$$

**Matrix \( $V^{T}$ \):**

$$
V^{T}_{\text{black box}} = \begin{bmatrix}
-0.59641821 & 0.26271876 & 0.75846171 \\
-0.77236466 & 0.06937103 & -0.63137983 \\
-0.2184906 & -0.96237545 & 0.16154054 
\end{bmatrix}
$$

**Condition Number:** \( Κ = 331.9745145809871 \)
**Condition Number:**  K<sub>black box</sub> = 331.974514579576

**Inverse of A:**

$$
A^{-1} = \begin{bmatrix}
-24 & 18 & 5 \\
20 & -15 & -4 \\
-5 & 4 & 1 
\end{bmatrix}
$$

$$
A^{-1}_{\text{black box}} = \begin{bmatrix}
-24 & 18 & 5 \\
20 & -15 & -4 \\
-5 & 4 & 1 
\end{bmatrix}
$$

**If the matrix is noninvertable**
Using the Matrix: 
$$
\begin{bmatrix}
2 & 4 \\
1 & 2 \\
\end{bmatrix}
$$

you'll get the following message: 

```plaintext
Traceback (most recent call last):  
  File "c:\Users\jeet\Desktop\COE 352 Project1.py", line 67, in <module>  
    u_custom, sigma_custom, v_custom, singular_values_custom, cond_custom, A_inv_custom, eig_vals_custom = svd_custom(B)  
  File "c:\Users\jeet\Desktop\COE 352 Project1.py", line 34, in svd_custom  
    raise ValueError("The matrix is non-invertible (has zero singular values).")  
ValueError: The matrix is non-invertible (has zero singular values)
```

# Fixed-Free System Example
For the Fixed-Free system example we use 3 masses, with 3 springs, where masses = spring constants = 1 we get the result: 
![image](https://github.com/user-attachments/assets/0e749c25-fe90-4143-a70b-c0afaf6b7137)

