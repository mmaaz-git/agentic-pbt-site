#!/usr/bin/env python3

import math
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from pandas.io.parsers.readers import _validate_names

@given(st.lists(st.floats(allow_nan=True), min_size=2, max_size=10))
def test_validate_names_detects_nan_duplicates(names):
    nan_count = sum(1 for x in names if isinstance(x, float) and math.isnan(x))
    if nan_count > 1:
        try:
            _validate_names(names)
            # If we got here, the function didn't reject duplicate NaNs
            print(f"Failed on input: {names} (contains {nan_count} NaN values)")
            assert False, f"Should reject duplicate NaN in {names}"
        except ValueError:
            # This is expected - duplicate NaNs should be rejected
            pass

# Run the test
print("Running hypothesis test...")
try:
    test_validate_names_detects_nan_duplicates()
    print("Test completed without finding any issues")
except AssertionError as e:
    print(f"Test found a bug: {e}")