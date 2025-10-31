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
    print(f"Input u: {u}")
    print(f"Input v: {v}")
    print(f"Distance: {d}")
    assert 0.0 <= d <= 1.0 + 1e-9

# Test the specific failing input manually
if __name__ == "__main__":
    # Test with the specific failing input
    u = np.array([0.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 0.0])
    d = braycurtis(u, v)
    print(f"braycurtis([0, 0, 0], [0, 0, 0]) = {d}")
    print(f"Is NaN: {np.isnan(d)}")

    # Run the hypothesis test
    try:
        test_braycurtis_range_positive_inputs()
    except AssertionError as e:
        print(f"Hypothesis test failed: {e}")