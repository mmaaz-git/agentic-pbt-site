import numpy as np
import numpy.random as npr
from hypothesis import given, strategies as st


@given(st.integers(min_value=2, max_value=10))
def test_dirichlet_all_zeros_violates_simplex_constraint(size):
    rng = npr.default_rng(42)

    alpha = np.zeros(size)
    result = rng.dirichlet(alpha)

    assert np.isclose(result.sum(), 1.0)

if __name__ == "__main__":
    test_dirichlet_all_zeros_violates_simplex_constraint()