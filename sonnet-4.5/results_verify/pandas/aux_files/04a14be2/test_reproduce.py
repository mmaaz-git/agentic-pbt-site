import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.io.sas.sas7bdat import _parse_datetime
from hypothesis import given, strategies as st, assume

# First, let's test if the function exists and works for basic cases
print("Testing basic functionality:")
result1 = _parse_datetime(0, "s")
print(f"_parse_datetime(0, 's') = {result1}")

result2 = _parse_datetime(86400, "s")  # 1 day in seconds
print(f"_parse_datetime(86400, 's') = {result2}")

result3 = _parse_datetime(1, "d")
print(f"_parse_datetime(1, 'd') = {result3}")

# Test NaN handling
import numpy as np
result4 = _parse_datetime(float('nan'), "s")
print(f"_parse_datetime(nan, 's') = {result4}")

print("\n" + "="*50)
print("Testing the reported OverflowError:")
print("="*50)

# Test the reported bug with large values
try:
    result = _parse_datetime(253717920000.0, "s")
    print(f"_parse_datetime(253717920000.0, 's') = {result}")
except OverflowError as e:
    print(f"OverflowError with seconds: {e}")

try:
    result = _parse_datetime(2936550.0, "d")
    print(f"_parse_datetime(2936550.0, 'd') = {result}")
except OverflowError as e:
    print(f"OverflowError with days: {e}")

# Test the hypothesis test
print("\n" + "="*50)
print("Testing the hypothesis test case:")
print("="*50)

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_parse_datetime_seconds_increases_monotonically(sas_datetime):
    assume(sas_datetime >= 0)
    assume(sas_datetime < 1e15)

    dt1 = _parse_datetime(sas_datetime, "s")
    dt2 = _parse_datetime(sas_datetime + 1, "s")

    if not pd.isna(dt1) and not pd.isna(dt2):
        assert dt2 > dt1, f"Failed for sas_datetime={sas_datetime}"

# Run a simple test with the failing input
sas_datetime = 253717919999.0
try:
    dt1 = _parse_datetime(sas_datetime, "s")
    dt2 = _parse_datetime(sas_datetime + 1, "s")
    print(f"dt1 = {dt1}")
    print(f"dt2 = {dt2}")
    if not pd.isna(dt1) and not pd.isna(dt2):
        print(f"dt2 > dt1: {dt2 > dt1}")
except OverflowError as e:
    print(f"OverflowError at sas_datetime={sas_datetime}: {e}")