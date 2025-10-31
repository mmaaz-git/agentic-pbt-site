import numpy as np
from scipy.optimize import least_squares

def test_specific_input():
    """Test the specific failing input from the bug report."""
    x0_list = [0.0, 0.0, 1.2751402521491925e-80]
    x0 = np.array(x0_list)
    n = len(x0)

    np.random.seed(hash(tuple(x0_list)) % (2**32))
    A = np.random.randn(n + 2, n)
    b = np.random.randn(n + 2)

    def residual_func(x):
        return A @ x - b

    def jacobian_func(x):
        return A

    result = least_squares(residual_func, x0, jac=jacobian_func, method='lm')

    if result.success:
        final_residual = residual_func(result.x)
        jac = jacobian_func(result.x)
        gradient = jac.T @ final_residual
        gradient_norm = np.linalg.norm(gradient)

        print(f"Result with x0 near zero:")
        print(f"  success: {result.success}")
        print(f"  message: {result.message}")
        print(f"  optimality: {result.optimality}")
        print(f"  gradient norm: {gradient_norm}")
        print(f"  x: {result.x}")
        print(f"  x0: {x0}")
        print(f"  x changed: {not np.allclose(result.x, x0)}")

        if gradient_norm >= 1e-6:
            print(f"\nAssertion would fail: Gradient J^T r should be near zero at optimum, got norm {gradient_norm}")
        else:
            print("\nAssertion would pass: Gradient is near zero")

test_specific_input()