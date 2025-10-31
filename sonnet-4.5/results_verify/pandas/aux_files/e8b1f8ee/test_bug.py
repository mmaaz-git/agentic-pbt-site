#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages")

# Test 1: Run the hypothesis test
from hypothesis import given, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

@given(st.just(None))
def test_argsort_defaults_kind_value(dummy):
    assert ARGSORT_DEFAULTS["kind"] is None

print("Running hypothesis test...")
test_argsort_defaults_kind_value()
print("Hypothesis test passed!")

# Test 2: Run the simple reproduction
print("\nReproducing the bug:")
print("ARGSORT_DEFAULTS:", ARGSORT_DEFAULTS)
print(f"ARGSORT_DEFAULTS['kind'] = {ARGSORT_DEFAULTS['kind']}")

assert ARGSORT_DEFAULTS["kind"] is None
print("Assertion passed: ARGSORT_DEFAULTS['kind'] is None")

# Show all keys and values
print("\nAll ARGSORT_DEFAULTS entries:")
for key, value in ARGSORT_DEFAULTS.items():
    print(f"  {key}: {value}")