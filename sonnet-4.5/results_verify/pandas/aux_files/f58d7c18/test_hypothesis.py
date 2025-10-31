from hypothesis import given, strategies as st
import pandas as pd
from pandas import RangeIndex
import numpy as np

@given(
    st.integers(min_value=-1000, max_value=1000),
    st.integers(min_value=-1000, max_value=1000),
    st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0)
)
def test_rangeindex_shallow_copy_with_equally_spaced_values(start, stop, step):
    """RangeIndex._shallow_copy should return RangeIndex for equally spaced values."""
    ri = RangeIndex(start, stop, step)
    if len(ri) == 0:
        return

    values = np.array(list(ri))
    result = ri._shallow_copy(values)

    # BUG: Fails for single-element ranges
    assert isinstance(result, RangeIndex), \
        f"Expected RangeIndex for equally-spaced values, got {type(result)}"

if __name__ == "__main__":
    # Test with the specific failing input
    test_rangeindex_shallow_copy_with_equally_spaced_values(0, 1, 1)
    print("Test passed with specific input")

    # Run full hypothesis test
    test_rangeindex_shallow_copy_with_equally_spaced_values()
    print("Hypothesis test completed")