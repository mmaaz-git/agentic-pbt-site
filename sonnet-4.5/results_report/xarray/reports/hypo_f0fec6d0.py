from hypothesis import given, strategies as st, assume
import numpy as np
from xarray.indexes import RangeIndex

@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    num=st.integers(min_value=2, max_value=1000),
)
def test_linspace_endpoint_true_last_value_equals_stop(start, stop, num):
    assume(abs(stop - start) > 1e-6)

    index = RangeIndex.linspace(start, stop, num, endpoint=True, dim="x")
    values = index.transform.forward({"x": np.arange(num)})["x"]

    assert values[-1] == stop

if __name__ == "__main__":
    # Run the test
    test_linspace_endpoint_true_last_value_equals_stop()