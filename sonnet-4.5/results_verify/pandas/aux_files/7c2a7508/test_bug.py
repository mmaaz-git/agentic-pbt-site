#!/usr/bin/env python3

import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.io.sas.sas7bdat import _convert_datetimes

# First, let's try the exact failing case mentioned
print("Testing exact failing case: 1e20")
try:
    series = pd.Series([1e20])
    result = _convert_datetimes(series, 's')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Now let's run the hypothesis test
print("Running hypothesis test:")

@given(value=st.floats(min_value=1e15, max_value=1e20, allow_nan=False, allow_infinity=False))
@settings(max_examples=50)
def test_convert_datetimes_extreme_values_seconds(value):
    series = pd.Series([value])
    try:
        result = _convert_datetimes(series, 's')
        print(f"Value {value}: Success, result = {result.iloc[0]}")
        assert len(result) == 1
    except OverflowError as e:
        print(f"Value {value}: OverflowError - {e}")
        raise
    except Exception as e:
        print(f"Value {value}: Other error - {type(e).__name__}: {e}")
        raise

try:
    test_convert_datetimes_extreme_values_seconds()
    print("\nAll hypothesis tests passed!")
except Exception as e:
    print(f"\nHypothesis test failed!")

# Let's also test some boundary cases
print("\n" + "="*50 + "\n")
print("Testing various large values:")

test_values = [1e10, 1e12, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20]
for val in test_values:
    try:
        series = pd.Series([val])
        result = _convert_datetimes(series, 's')
        print(f"Value {val}: Success, result = {result.iloc[0]}")
    except OverflowError as e:
        print(f"Value {val}: OverflowError - {e}")
    except Exception as e:
        print(f"Value {val}: Other error - {type(e).__name__}: {e}")