import numpy as np
import scipy.spatial.distance as distance
from hypothesis import given, strategies as st, assume
import math


@given(
    st.lists(st.integers(min_value=0, max_value=1), min_size=5, max_size=50),
    st.lists(st.integers(min_value=0, max_value=1), min_size=5, max_size=50)
)
def test_dice_bounds(u, v):
    assume(len(u) == len(v))
    u_arr = np.array(u, dtype=bool)
    v_arr = np.array(v, dtype=bool)

    d = distance.dice(u_arr, v_arr)

    assert 0 <= d <= 1 or math.isnan(d), f"Dice dissimilarity should be in [0,1] or NaN, got {d}"
    if not math.isnan(d):
        assert 0 <= d <= 1, f"Dice dissimilarity should be in [0,1], got {d}"

# Manual test of the failing case
if __name__ == "__main__":
    u = np.array([0, 0, 0, 0, 0], dtype=bool)
    v = np.array([0, 0, 0, 0, 0], dtype=bool)

    d = distance.dice(u, v)
    print(f"dice([0,0,0,0,0], [0,0,0,0,0]) = {d}")

    if math.isnan(d):
        print("Result is NaN - BUG CONFIRMED")
    else:
        print(f"Result is {d}")

    # Run hypothesis test
    test_dice_bounds()