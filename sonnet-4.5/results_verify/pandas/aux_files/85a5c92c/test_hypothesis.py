#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st
from pandas.api import types
import re

@given(st.text(min_size=1, max_size=100))
def test_is_re_compilable_should_not_raise(pattern_str):
    """Property-based test: is_re_compilable should never raise an exception"""
    result = types.is_re_compilable(pattern_str)

    assert isinstance(result, bool), f"Result should be bool, got {type(result)}"

    # If it returns True, we should be able to compile it
    if result:
        re.compile(pattern_str)

# Run the test
if __name__ == "__main__":
    print("Running hypothesis property-based test...")
    try:
        test_is_re_compilable_should_not_raise()
        print("ERROR: Test passed (shouldn't happen if bug exists)")
    except Exception as e:
        print(f"Test failed as expected: {type(e).__name__}: {e}")
        print("\nTrying specific failure case: '?'")
        try:
            test_is_re_compilable_should_not_raise('?')
        except Exception as e2:
            print(f"Specific test failed: {type(e2).__name__}: {e2}")