#!/usr/bin/env python3

# First, let's run the property-based test
from hypothesis import given, strategies as st
import pytest
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.plotting._misc import _Options

def test_options_pop_protects_default_keys(value):
    opts = _Options()
    opts["x_compat"] = value

    with pytest.raises(ValueError, match="Cannot remove default parameter"):
        opts.pop("xaxis.compat")

# Run the property test
print("Running property-based test...")
try:
    test_options_pop_protects_default_keys(False)
    print("Test PASSED (unexpected - the test should fail)")
except AssertionError as e:
    print(f"Test FAILED (as expected): {e}")
except Exception as e:
    print(f"Test error: {e}")

print("\n" + "="*60)
print("Reproducing the bug manually...")
print("="*60 + "\n")

# Now reproduce the bug manually
opts = _Options()
print(f"Initial state: {dict(opts)}")

removed_value = opts.pop("xaxis.compat")
print(f"Removed value: {removed_value}")
print(f"State after pop: {dict(opts)}")

print("\nComparison - del correctly raises error:")
opts2 = _Options()
try:
    del opts2["xaxis.compat"]
    print("del succeeded (unexpected)")
except ValueError as e:
    print(f"del raised ValueError: {e}")