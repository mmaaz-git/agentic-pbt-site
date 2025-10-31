import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import assume, given, strategies as st
from xarray.indexes import RangeIndex


@given(
    start=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    step=st.floats(min_value=0.01, max_value=10, allow_nan=False, allow_infinity=False),
)
def test_arange_matches_numpy_semantics(start, stop, step):
    assume(stop > start)

    idx = RangeIndex.arange(start, stop, step, dim="x")
    np_range = np.arange(start, stop, step)

    assert idx.size == len(np_range), f"Size mismatch: RangeIndex={idx.size}, NumPy={len(np_range)}"

    xr_values = idx.transform.forward({"x": np.arange(idx.size)})["x"]

    try:
        assert np.allclose(xr_values, np_range), f"Value mismatch: RangeIndex={xr_values}, NumPy={np_range}"
    except AssertionError as e:
        # Re-raise with more detail
        print(f"Failed for: start={start}, stop={stop}, step={step}")
        print(f"RangeIndex values: {xr_values}")
        print(f"NumPy values: {np_range}")
        print(f"RangeIndex step: {idx.step}")
        print(f"Expected step: {step}")
        raise

if __name__ == "__main__":
    # Run the test
    test_arange_matches_numpy_semantics()