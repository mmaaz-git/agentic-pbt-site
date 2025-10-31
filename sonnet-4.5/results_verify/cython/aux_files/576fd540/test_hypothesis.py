#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.dtypes.common import ensure_python_int
import pytest

@given(st.floats(allow_nan=True, allow_infinity=True))
@settings(max_examples=200)
def test_ensure_python_int_special_floats(value):
    if np.isnan(value) or np.isinf(value):
        # The test expects TypeError to be raised
        try:
            result = ensure_python_int(value)
            print(f"ERROR: Expected exception for {value}, but got result: {result}")
        except TypeError:
            pass  # Expected
        except OverflowError as e:
            print(f"ERROR: Got OverflowError for {value} instead of TypeError: {e}")
        except Exception as e:
            print(f"ERROR: Got unexpected exception for {value}: {type(e).__name__}: {e}")
    elif value == int(value):
        try:
            result = ensure_python_int(value)
            assert result == int(value)
        except Exception as e:
            print(f"ERROR: Unexpected exception for valid value {value}: {type(e).__name__}: {e}")

# Run the test
if __name__ == "__main__":
    print("Running hypothesis test...")
    test_ensure_python_int_special_floats()
    print("\nTest with specific failing input: float('inf')")
    try:
        test_ensure_python_int_special_floats(float('inf'))
    except Exception as e:
        print(f"Exception: {e}")