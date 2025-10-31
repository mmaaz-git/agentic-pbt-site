import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple
from hypothesis import given, strategies as st, settings
import pytest


@given(
    axis=st.integers(min_value=2**31, max_value=2**40),
    ndim=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100)
def test_large_positive_axis_raises_axis_error(axis, ndim):
    with pytest.raises(np.exceptions.AxisError):
        normalize_axis_tuple(axis, ndim)

# Run the test
if __name__ == "__main__":
    test_large_positive_axis_raises_axis_error()