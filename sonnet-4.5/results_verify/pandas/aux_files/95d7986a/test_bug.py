import pandas as pd
from pandas.io.sas.sas7bdat import _convert_datetimes
import numpy as np
import traceback

print("Testing _convert_datetimes with edge cases")
print("=" * 50)

# Test 1: Normal value
print("\nTest 1: Normal datetime value (100 seconds from SAS origin)")
try:
    series = pd.Series([100.0])
    result = _convert_datetimes(series, "s")
    print(f"Input: {series.values}")
    print(f"Result: {result}")
    print(f"Result type: {type(result.iloc[0])}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

# Test 2: Very large value (1e16)
print("\nTest 2: Very large value (1e16 seconds)")
try:
    series = pd.Series([1e16])
    result = _convert_datetimes(series, "s")
    print(f"Input: {series.values}")
    print(f"Result: {result}")
    print(f"Result type: {type(result.iloc[0])}")
except Exception as e:
    print(f"ERROR: {e}")
    print(f"Exception type: {type(e).__name__}")

# Test 3: Even larger value
print("\nTest 3: Even larger value (9.223372036854776e+18)")
try:
    series = pd.Series([9.223372036854776e+18])
    result = _convert_datetimes(series, "s")
    print(f"Input: {series.values}")
    print(f"Result: {result}")
    print(f"Result type: {type(result.iloc[0])}")
except Exception as e:
    print(f"ERROR: {e}")
    print(f"Exception type: {type(e).__name__}")

# Test 4: NaN value (should be handled)
print("\nTest 4: NaN value")
try:
    series = pd.Series([np.nan])
    result = _convert_datetimes(series, "s")
    print(f"Input: {series.values}")
    print(f"Result: {result}")
    print(f"Is NaT: {pd.isna(result.iloc[0])}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

# Test 5: Mixed values
print("\nTest 5: Mixed values (normal, large, NaN)")
try:
    series = pd.Series([100.0, 1e16, np.nan])
    result = _convert_datetimes(series, "s")
    print(f"Input: {series.values}")
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {e}")
    print(f"Exception type: {type(e).__name__}")

# Test 6: Test with unit='d' (days)
print("\nTest 6: Large value with unit='d' (1e10 days)")
try:
    series = pd.Series([1e10])
    result = _convert_datetimes(series, "d")
    print(f"Input: {series.values}")
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {e}")
    print(f"Exception type: {type(e).__name__}")

# Test the hypothesis test
print("\n" + "=" * 50)
print("Running the Hypothesis test from bug report:")
print("=" * 50)

from hypothesis import given, strategies as st, settings

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                          min_value=1e16, max_value=1e18), min_size=1, max_size=10))
@settings(max_examples=10)  # Reduced for demonstration
def test_convert_datetimes_very_large_should_not_crash(values):
    series = pd.Series(values)
    try:
        result = _convert_datetimes(series, "s")
        assert isinstance(result, pd.Series)
        print(f"✓ Passed for values starting with {values[0]:.2e}")
    except Exception as e:
        print(f"✗ Failed for values starting with {values[0]:.2e}: {type(e).__name__}: {e}")
        raise

try:
    test_convert_datetimes_very_large_should_not_crash()
    print("All hypothesis tests passed!")
except Exception as e:
    print(f"Hypothesis test failed!")