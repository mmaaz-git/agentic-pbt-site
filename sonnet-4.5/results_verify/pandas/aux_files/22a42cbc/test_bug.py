#!/usr/bin/env python3
import sys

# Test the hypothesis test
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

def test_argsort_defaults_kind_consistency():
    """
    Property: Dictionary values should not be overwritten without reason.
    The final value should be the only assignment.
    """
    assert 'kind' in ARGSORT_DEFAULTS
    assert ARGSORT_DEFAULTS['kind'] is None
    print("Test passed: ARGSORT_DEFAULTS['kind'] is None")
    return True

# Run the test
test_argsort_defaults_kind_consistency()

# Reproduce the bug
with open('/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/compat/numpy/function.py', 'r') as f:
    lines = f.readlines()
    print("Line 138:", lines[137].strip())
    print("Line 140:", lines[139].strip())

print(f"Final value: ARGSORT_DEFAULTS['kind'] = {ARGSORT_DEFAULTS['kind']}")