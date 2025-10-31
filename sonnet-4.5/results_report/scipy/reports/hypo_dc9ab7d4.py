import numpy as np
from hypothesis import given, strategies as st
from scipy.differentiate import jacobian


@given(st.integers(min_value=2, max_value=5))
def test_jacobian_of_linear_function(m):
    rng = np.random.default_rng()
    A = rng.standard_normal((m, m))

    def f(xi):
        return A @ xi

    x = rng.standard_normal(m)
    res = jacobian(f, x)

    if res.success.all():
        assert np.allclose(res.df, A, rtol=1e-4), \
            f"Jacobian should equal A, but got A.T"


if __name__ == "__main__":
    test_jacobian_of_linear_function()