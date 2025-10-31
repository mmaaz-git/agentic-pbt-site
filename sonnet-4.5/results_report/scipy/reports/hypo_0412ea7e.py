import numpy as np
from scipy.differentiate import jacobian
from hypothesis import given, strategies as st, settings

@given(size=st.integers(min_value=2, max_value=5))
@settings(max_examples=50)
def test_jacobian_linear_function(size):
    rng = np.random.RandomState(42)
    A = rng.randn(size, size)

    def f(x):
        return A @ x

    x = rng.randn(size)
    res = jacobian(f, x)

    if np.all(res.success):
        assert np.allclose(res.df, A, rtol=1e-4, atol=1e-6), \
            f"Jacobian mismatch: got {res.df}, expected {A}"

if __name__ == "__main__":
    test_jacobian_linear_function()