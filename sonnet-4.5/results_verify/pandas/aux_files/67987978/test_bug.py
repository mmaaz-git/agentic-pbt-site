#!/usr/bin/env python3

# First test the hypothesis test case
from hypothesis import given, strategies as st, settings
from pandas.io.sas.sas7bdat import _parse_datetime
from datetime import datetime
import pandas as pd

print("Testing with Hypothesis strategy...")

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
@settings(max_examples=1000)
def test_parse_datetime_days_unit(sas_datetime):
    try:
        result = _parse_datetime(sas_datetime, unit='d')
        if not pd.isna(result):
            assert isinstance(result, datetime)
        print(f"✓ Passed for sas_datetime={sas_datetime}")
    except Exception as e:
        print(f"✗ Failed for sas_datetime={sas_datetime}: {e}")
        raise

# Run the hypothesis test
try:
    test_parse_datetime_days_unit()
    print("All hypothesis tests passed")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

print("\n" + "="*60)
print("Testing specific failing input...")
print("="*60 + "\n")

# Now test the specific failing case mentioned
sas_datetime = 2936550.0
print(f"Testing sas_datetime={sas_datetime}, unit='d'")

try:
    result = _parse_datetime(sas_datetime, unit='d')
    print(f"Result: {result}")
    print(f"Type: {type(result)}")
except OverflowError as e:
    print(f"OverflowError occurred: {e}")
except Exception as e:
    print(f"Other error occurred: {type(e).__name__}: {e}")

# Test some edge cases
print("\n" + "="*60)
print("Testing edge cases...")
print("="*60 + "\n")

test_cases = [
    (2936549.0, "Just before overflow"),
    (2936550.0, "At overflow boundary"),
    (-2936550.0, "Large negative value"),
    (float('nan'), "NaN value"),
]

for test_val, description in test_cases:
    print(f"\nTesting {description}: sas_datetime={test_val}")
    try:
        result = _parse_datetime(test_val, unit='d')
        print(f"  Result: {result}")
        if not pd.isna(result):
            print(f"  Date: {result.date()}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")