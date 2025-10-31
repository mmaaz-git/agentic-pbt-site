import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from pandas.core.array_algos.putmask import putmask_without_repeat


@given(
    values_size=st.integers(min_value=10, max_value=50),
    new_size=st.integers(min_value=1, max_value=9)
)
@settings(max_examples=300)
def test_putmask_without_repeat_length_mismatch_error(values_size, new_size):
    assume(new_size != values_size)

    values = np.arange(values_size)
    mask = np.ones(values_size, dtype=bool)
    new = np.arange(new_size)

    with pytest.raises(ValueError, match="cannot assign mismatch"):
        putmask_without_repeat(values, mask, new)

# Test the failing case specifically
if __name__ == "__main__":
    print("Testing the hypothesis test with the failing input:")
    values_size = 10
    new_size = 1

    values = np.arange(values_size)
    mask = np.ones(values_size, dtype=bool)
    new = np.arange(new_size)

    print(f"values: {values}")
    print(f"mask: {mask}")
    print(f"new: {new}")

    try:
        putmask_without_repeat(values, mask, new)
        print(f"Result values: {values}")
        print("ERROR: No exception was raised!")
    except ValueError as e:
        print(f"Exception raised as expected: {e}")