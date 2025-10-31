import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import assume, given, strategies as st, example
from xarray.indexes import RangeIndex


@given(
    start=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    step=st.floats(min_value=0.01, max_value=10, allow_nan=False, allow_infinity=False),
)
@example(start=-1.5, stop=0.0, step=1.0)  # Add the specific failing case
def test_arange_matches_numpy_semantics(start, stop, step):
    assume(stop > start)

    idx = RangeIndex.arange(start, stop, step, dim="x")
    np_range = np.arange(start, stop, step)

    assert idx.size == len(np_range), f"Size mismatch: RangeIndex={idx.size}, NumPy={len(np_range)}"

    xr_values = idx.transform.forward({"x": np.arange(idx.size)})["x"]
    assert np.allclose(xr_values, np_range), f"Value mismatch: RangeIndex={xr_values}, NumPy={np_range}"

# Run the test
if __name__ == "__main__":
    test_arange_matches_numpy_semantics()