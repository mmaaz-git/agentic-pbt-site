#!/usr/bin/env python3
"""Test the Hypothesis property-based test from the bug report."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from xarray.backends.netcdf3 import is_valid_nc3_name

@given(st.text())
def test_is_valid_nc3_name_doesnt_crash(s):
    """The function should never crash, always return a bool."""
    result = is_valid_nc3_name(s)
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"

# Run the test
if __name__ == "__main__":
    print("Running Hypothesis test...")
    try:
        test_is_valid_nc3_name_doesnt_crash()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed with: {e}")

    # Test specifically with empty string
    print("\nTesting specifically with empty string:")
    try:
        test_is_valid_nc3_name_doesnt_crash.hypothesis.inner_test("")
        print("Empty string test passed!")
    except Exception as e:
        print(f"Empty string test failed with: {e}")
        import traceback
        traceback.print_exc()