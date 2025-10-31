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
        f"Expected RangeIndex for equally-spaced values, got {type(result).__name__} for values={values}"

# Run the test
if __name__ == "__main__":
    test_rangeindex_shallow_copy_with_equally_spaced_values()