from hypothesis import given, strategies as st, assume
import numpy as np
from scipy.optimize import least_squares

@given(
    st.lists(st.floats(min_value=-3, max_value=3, allow_nan=False, allow_infinity=False),
             min_size=3, max_size=6)
)
def test_least_squares_gradient_zero_at_optimum(x0_list):
    """Test that at optimum, gradient J^T r is near zero."""
    x0 = np.array(x0_list)
    assume(np.linalg.norm(x0) < 10)
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

        assert gradient_norm < 1e-6, \
            f"Gradient J^T r should be near zero at optimum, got norm {gradient_norm}"

# Test with the specific failing input
print("Testing with specific failing input...")
try:
    test_least_squares_gradient_zero_at_optimum([0.0, 0.0, 1.2751402521491925e-80])
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")