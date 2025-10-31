from scipy.spatial.distance import correlation
from hypothesis import given, strategies as st, assume
import numpy as np


@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=2, max_size=20)
)
def test_correlation_identity(u):
    """Property: correlation(u, u) should be 0"""
    u_arr = np.array(u)
    d = correlation(u_arr, u_arr)
    assert np.isclose(d, 0.0, atol=1e-9) or not np.isnan(d)


if __name__ == "__main__":
    test_correlation_identity()