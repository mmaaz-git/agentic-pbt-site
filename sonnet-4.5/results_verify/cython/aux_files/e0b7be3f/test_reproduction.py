#!/usr/bin/env python3
"""Test script to reproduce the Field.__repr__ bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

# First test the simple reproduction
from Cython.Utility import field

print("Testing simple reproduction:")
f = field(kw_only=True)
repr_str = repr(f)
print(f"repr(field(kw_only=True)): {repr_str}")
print(f"Checking if 'kw_only=True' in repr: {'kw_only=True' in repr_str}")
print(f"Checking if 'kwonly=True' in repr: {'kwonly=True' in repr_str}")
print()

# Now test with hypothesis
from hypothesis import given, strategies as st

@given(st.booleans())
def test_field_repr_consistency(kw_only_value):
    f = field(kw_only=kw_only_value)
    repr_str = repr(f)
    # The test from the bug report
    assert f'kw_only={kw_only_value!r}' in repr_str or f'kwonly={kw_only_value!r}' in repr_str
    # More detailed check
    if f'kw_only={kw_only_value!r}' in repr_str:
        print(f"✓ repr correctly uses 'kw_only' for value {kw_only_value}")
    elif f'kwonly={kw_only_value!r}' in repr_str:
        print(f"✗ repr incorrectly uses 'kwonly' for value {kw_only_value}")
    else:
        print(f"? Neither format found for value {kw_only_value}")
    return repr_str

print("Testing with Hypothesis:")
try:
    test_field_repr_consistency()
    print("Hypothesis test passed (but likely only because of the 'or' condition)")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

# Let's also check what the actual attribute name is
print("\nChecking actual attribute:")
f = field(kw_only=True)
print(f"f.kw_only = {f.kw_only}")
try:
    print(f"f.kwonly = {f.kwonly}")
except AttributeError as e:
    print(f"f.kwonly raises AttributeError: {e}")