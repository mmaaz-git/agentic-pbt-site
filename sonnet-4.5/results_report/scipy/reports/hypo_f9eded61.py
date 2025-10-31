import numpy as np
import scipy.spatial.distance as distance
from hypothesis import given, strategies as st, assume


@given(
    st.lists(st.integers(min_value=0, max_value=1), min_size=5, max_size=50),
    st.lists(st.integers(min_value=0, max_value=1), min_size=5, max_size=50)
)
def test_dice_bounds(u, v):
    assume(len(u) == len(v))
    u_arr = np.array(u, dtype=bool)
    v_arr = np.array(v, dtype=bool)

    d = distance.dice(u_arr, v_arr)

    assert 0 <= d <= 1, f"Dice dissimilarity should be in [0,1], got {d}"


if __name__ == "__main__":
    # Run the property-based test
    test_dice_bounds()