#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, settings
from pandas.core.dtypes import inference
import re

# Test from the bug report
@given(st.text(min_size=0, max_size=100))
@settings(max_examples=50, deadline=None)
def test_is_re_compilable_on_regex_patterns(pattern):
    try:
        re.compile(pattern)
        result = inference.is_re_compilable(pattern)
        assert result is True, f"Expected True for compilable pattern: {pattern!r}"
    except re.error:
        # This pattern cannot be compiled, so is_re_compilable should return False
        try:
            result = inference.is_re_compilable(pattern)
            assert result is False, f"Expected False for non-compilable pattern: {pattern!r}"
        except Exception as e:
            raise AssertionError(f"is_re_compilable raised {type(e).__name__} for invalid pattern {pattern!r}: {e}")

# Run the test
print("Running hypothesis test...")
try:
    test_is_re_compilable_on_regex_patterns()
    print("Test passed!")
except Exception as e:
    print(f"Test failed: {e}")

# Also test the specific failing case mentioned
print("\nTesting specific case: ')'")
try:
    re.compile(")")
    print("  ')' is compilable")
except re.error as e:
    print(f"  ')' is NOT compilable: {e}")

try:
    result = inference.is_re_compilable(")")
    print(f"  is_re_compilable(')') returned: {result}")
except Exception as e:
    print(f"  is_re_compilable(')') raised: {type(e).__name__}: {e}")