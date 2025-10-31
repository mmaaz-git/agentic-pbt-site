#!/usr/bin/env python3
"""Property-based test that discovers the ZeroDivisionError bug in RangeIndex.linspace"""

from hypothesis import given, strategies as st, assume, settings
from xarray.indexes import RangeIndex

@given(
    start=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
    endpoint=st.booleans()
)
@settings(max_examples=1000)
def test_linspace_with_num_1(start, stop, endpoint):
    assume(start != stop)
    index = RangeIndex.linspace(start, stop, num=1, endpoint=endpoint, dim="x")
    assert index.size == 1

if __name__ == "__main__":
    # Run the test to find a failure
    test_linspace_with_num_1()
    print("Test passed!")  # This will not be reached if the test fails