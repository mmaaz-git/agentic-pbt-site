from hypothesis import given, strategies as st, assume, settings
import numpy as np
import pytest
from pandas.api.extensions import take

# The exact test from the bug report
@settings(max_examples=500)
@given(
    size=st.integers(min_value=1, max_value=100),
    idx=st.integers()
)
def test_take_out_of_bounds_negative(size, idx):
    arr = np.arange(size)
    assume(idx < -size)

    with pytest.raises(IndexError):
        take(arr, [idx])

# Run the specific failing case
print("Testing specific failing case: size=1, idx=-9_223_372_036_854_775_809")
try:
    test_take_out_of_bounds_negative(size=1, idx=-9_223_372_036_854_775_809)
    print("Test passed with specific inputs")
except Exception as e:
    print(f"Test failed: {e}")

# Also test boundary cases
print("\nTesting C long boundary cases:")

# Test at the C long min boundary
import sys
C_LONG_MIN = -(2**63)  # For 64-bit systems
C_LONG_MAX = 2**63 - 1

test_cases = [
    (1, C_LONG_MIN - 1, "Below C_LONG_MIN"),
    (1, C_LONG_MIN, "At C_LONG_MIN"),
    (1, -sys.maxsize - 2, "Below sys.maxsize"),
]

for size, idx, desc in test_cases:
    print(f"\n{desc}: size={size}, idx={idx}")
    arr = np.arange(size)
    try:
        result = take(arr, [idx])
        print(f"  No error (unexpected)")
    except OverflowError as e:
        print(f"  OverflowError: {e}")
    except IndexError as e:
        print(f"  IndexError: {e}")