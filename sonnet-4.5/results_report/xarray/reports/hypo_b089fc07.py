from hypothesis import given, strategies as st, example
import numpy as np
from xarray.plot.utils import _infer_interval_breaks

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                          min_value=-1e6, max_value=1e6),
               min_size=2, max_size=100))
@example([0.0, -1.0])  # The minimal failing case
def test_infer_interval_breaks_covers_input_range(coord):
    coord_arr = np.array(coord)
    result = _infer_interval_breaks(coord_arr)

    # Property 1: First break should be <= minimum coordinate value
    assert result[0] <= coord_arr.min(), \
        f"First break should be <= min, got {result[0]} > {coord_arr.min()}"

    # Property 2: Last break should be >= maximum coordinate value
    assert result[-1] >= coord_arr.max(), \
        f"Last break should be >= max, got {result[-1]} < {coord_arr.max()}"

if __name__ == "__main__":
    # Run the test
    test_infer_interval_breaks_covers_input_range()