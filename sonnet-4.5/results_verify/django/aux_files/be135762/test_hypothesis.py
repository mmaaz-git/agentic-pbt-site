#!/usr/bin/env python3
"""Hypothesis test from the bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import pytest
from django.template import Variable

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=1, max_value=1e10))
def test_variable_float_with_trailing_dot_should_be_rejected(num):
    """
    Property: Floats with trailing dots should be rejected as invalid.
    Evidence: Code comment on line 824 says '"2." is invalid' and code
    explicitly raises ValueError for this case on line 826.
    """
    var_str = f"{int(num)}."

    with pytest.raises((ValueError, Exception)):
        var = Variable(var_str)

if __name__ == "__main__":
    # Run a few examples manually
    test_cases = [1.0, 2.0, 10.0, 999.0, 1234.0]

    for num in test_cases:
        var_str = f"{int(num)}."
        print(f"Testing '{var_str}'...")
        try:
            var = Variable(var_str)
            print(f"  ERROR: No exception raised! Created Variable with:")
            print(f"    literal: {var.literal}")
            print(f"    lookups: {var.lookups}")
        except (ValueError, Exception) as e:
            print(f"  OK: Exception raised: {type(e).__name__}: {e}")