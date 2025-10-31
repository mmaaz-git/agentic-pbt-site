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
        print(f"Testing with m={m}")
        print(f"Matrix A shape: {A.shape}")
        print(f"Jacobian shape: {res.df.shape}")
        print(f"A:\n{A}")
        print(f"res.df:\n{res.df}")
        print(f"A.T:\n{A.T}")
        print(f"Is res.df close to A? {np.allclose(res.df, A, rtol=1e-4)}")
        print(f"Is res.df close to A.T? {np.allclose(res.df, A.T, rtol=1e-4)}")
        assert np.allclose(res.df, A, rtol=1e-4), \
            f"Jacobian should equal A, but got A.T"

if __name__ == "__main__":
    test_jacobian_of_linear_function()