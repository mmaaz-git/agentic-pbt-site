from scipy.spatial.distance import dice
from hypothesis import given, strategies as st, assume
import numpy as np


@given(
    st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=30),
    st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=30)
)
def test_dice_distance_range(u, v):
    assume(len(u) == len(v))
    u_arr = np.array(u, dtype=bool)
    v_arr = np.array(v, dtype=bool)
    d = dice(u_arr, v_arr)
    assert 0.0 <= d <= 1.0 + 1e-9


if __name__ == "__main__":
    test_dice_distance_range()