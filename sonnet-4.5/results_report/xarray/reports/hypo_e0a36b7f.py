from hypothesis import given, strategies as st, settings, assume
from xarray.indexes import RangeIndex
import numpy as np

@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    step=st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=500)
def test_arange_step_matches_parameter(start, stop, step):
    assume(stop > start)
    assume(step > 0)
    assume((stop - start) / step < 1e6)

    index = RangeIndex.arange(start, stop, step, dim="x")

    coords = index.transform.forward({index.dim: np.arange(index.size)})
    values = coords[index.coord_name]

    if index.size > 1:
        actual_steps = np.diff(values)

        assert np.allclose(actual_steps, step, rtol=1e-9)

if __name__ == "__main__":
    test_arange_step_matches_parameter()