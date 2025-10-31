import numpy as np
from hypothesis import given, strategies as st, assume


@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=2, max_size=10))
def test_dirichlet_sums_to_one(alpha):
    assume(all(a > 0 for a in alpha))
    samples = np.random.dirichlet(alpha, size=100)
    sums = samples.sum(axis=1)
    assert np.allclose(sums, 1.0)


if __name__ == "__main__":
    test_dirichlet_sums_to_one()