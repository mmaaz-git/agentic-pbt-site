from scipy.spatial.distance import braycurtis
from hypothesis import given, strategies as st, assume
import numpy as np


@given(
    st.lists(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=30),
    st.lists(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=30)
)
def test_braycurtis_range_positive_inputs(u, v):
    assume(len(u) == len(v))
    u_arr = np.array(u)
    v_arr = np.array(v)
    d = braycurtis(u_arr, v_arr)
    assert 0.0 <= d <= 1.0 + 1e-9

if __name__ == "__main__":
    test_braycurtis_range_positive_inputs()