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

# Try different tolerance values
print("Testing with different tolerance values:\n")

for ftol_val in [1e-8, 1e-10, 1e-12]:
    for gtol_val in [1e-8, 1e-10, 1e-12]:
        try:
            result = least_squares(residual_func, x0, jac=jacobian_func, method='lm',
                                  ftol=ftol_val, gtol=gtol_val)
        except ValueError:
            continue

        gradient = jacobian_func(result.x).T @ residual_func(result.x)
        gradient_norm = np.linalg.norm(gradient)

        print(f"ftol={ftol_val}, gtol={gtol_val}:")
        print(f"  success: {result.success}, message: {result.message}")
        print(f"  optimality: {result.optimality:.6f}, gradient_norm: {gradient_norm:.6f}")
        print(f"  x changed: {not np.allclose(result.x, x0)}")
        print()

# Check what happens when we start from a slightly perturbed initial value
print("\nTesting with slightly perturbed initial values:")
for epsilon in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
    x0_perturbed = np.array([epsilon, epsilon, epsilon])
    result = least_squares(residual_func, x0_perturbed, jac=jacobian_func, method='lm')

    gradient = jacobian_func(result.x).T @ residual_func(result.x)
    gradient_norm = np.linalg.norm(gradient)

    print(f"x0 = [{epsilon}, {epsilon}, {epsilon}]:")
    print(f"  success: {result.success}, optimality: {result.optimality:.6f}")
    print(f"  gradient_norm: {gradient_norm:.6f}")
    print(f"  nfev: {result.nfev}, njev: {result.njev}")

# Let's check the cost function values
print("\nCost function analysis:")
x0 = np.array([0.0, 0.0, 1.2751402521491925e-80])
cost_at_x0 = 0.5 * np.sum(residual_func(x0)**2)
print(f"Cost at x0: {cost_at_x0}")

# Compute exact solution
x_exact, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
cost_at_exact = 0.5 * np.sum(residual_func(x_exact)**2)
print(f"Cost at exact solution: {cost_at_exact}")

print(f"Relative cost difference: {abs(cost_at_x0 - cost_at_exact) / cost_at_exact}")
print(f"ftol default: 1e-8")
print(f"Termination expected when: dF < ftol * F")