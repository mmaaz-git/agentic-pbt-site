#!/usr/bin/env python3
"""Property-based test for RangeIndex.linspace with num=1 and endpoint=True"""

from hypothesis import given, strategies as st, example
from xarray.indexes import RangeIndex


@given(
    start=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    stop=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
)
@example(start=0.0, stop=0.0)  # Explicitly test with both values equal
@example(start=0.0, stop=1.0)  # Standard case
def test_linspace_num_1_endpoint_true(start, stop):
    """Test that linspace works with num=1 and endpoint=True."""
    idx = RangeIndex.linspace(start, stop, num=1, endpoint=True, dim="x")
    assert idx.size == 1

    # Also test that we get a reasonable coordinate value
    coord_values = idx.transform.forward({idx.dim: [0]})
    assert idx.coord_name in coord_values
    assert len(coord_values[idx.coord_name]) == 1


if __name__ == "__main__":
    test_linspace_num_1_endpoint_true()