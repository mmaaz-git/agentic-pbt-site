import numpy as np
from scipy.optimize import least_squares

x0 = np.array([0.0, 0.0, 1.2751402521491925e-80])
n = len(x0)

np.random.seed(hash(tuple([0.0, 0.0, 1.2751402521491925e-80])) % (2**32))
A = np.random.randn(n + 2, n)
b = np.random.randn(n + 2)

def residual_func(x):
    return A @ x - b

def jacobian_func(x):
    return A

result = least_squares(residual_func, x0, jac=jacobian_func, method='lm')

print(f"success: {result.success}")
print(f"message: {result.message}")
print(f"x: {result.x}")
print(f"optimality: {result.optimality}")

gradient = jacobian_func(result.x).T @ residual_func(result.x)
gradient_norm = np.linalg.norm(gradient)
print(f"Gradient norm: {gradient_norm}")

x0_better = np.ones(n)
result2 = least_squares(residual_func, x0_better, jac=jacobian_func, method='lm')
gradient2_norm = np.linalg.norm(jacobian_func(result2.x).T @ residual_func(result2.x))

print(f"\nWith better initial guess:")
print(f"success: {result2.success}")
print(f"Gradient norm: {gradient2_norm}")

# Let's also check the actual solution quality
print(f"\nComparison of residual norms:")
print(f"Initial x0 near zero - residual norm: {np.linalg.norm(residual_func(result.x))}")
print(f"Initial x0 = ones - residual norm: {np.linalg.norm(residual_func(result2.x))}")

# Verify that this is a linear least squares problem and compute the exact solution
print(f"\nExact solution (using lstsq):")
x_exact, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
print(f"x_exact: {x_exact}")
print(f"Gradient norm at exact solution: {np.linalg.norm(A.T @ (A @ x_exact - b))}")
print(f"Residual norm at exact solution: {np.linalg.norm(A @ x_exact - b)}")