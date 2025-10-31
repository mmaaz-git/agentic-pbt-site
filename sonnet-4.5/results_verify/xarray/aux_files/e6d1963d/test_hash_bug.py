#!/usr/bin/env python3
"""Test CombineKwargDefault hash stability bug"""

from hypothesis import given, strategies as st, settings

# First, test the basic reproduction case
print("Testing basic reproduction case...")
import sys
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS

obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

# Save original option value
original_option = OPTIONS.get("use_new_combine_kwarg_defaults", False)

try:
    OPTIONS["use_new_combine_kwarg_defaults"] = False
    s = {obj}
    hash1 = hash(obj)
    print(f"Hash with option=False: {hash1}")
    assert obj in s, "Object should be found in set when option is False"
    print("✓ Object found in set with option=False")

    OPTIONS["use_new_combine_kwarg_defaults"] = True
    hash2 = hash(obj)
    print(f"Hash with option=True: {hash2}")

    # This should fail according to the bug report
    if obj in s:
        print("✓ Object found in set with option=True (BUG NOT REPRODUCED)")
    else:
        print("✗ Object NOT found in set with option=True (BUG REPRODUCED)")

    print(f"\nHash values are {'equal' if hash1 == hash2 else 'different'}")

finally:
    OPTIONS["use_new_combine_kwarg_defaults"] = original_option

# Now test with hypothesis
print("\n\nTesting with hypothesis...")

@given(st.sampled_from(["all", "minimal", "exact"]))
@settings(max_examples=10)
def test_hash_stability_across_options_change(val):
    obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

    original_option = OPTIONS.get("use_new_combine_kwarg_defaults", False)

    try:
        OPTIONS["use_new_combine_kwarg_defaults"] = False
        hash1 = hash(obj)

        OPTIONS["use_new_combine_kwarg_defaults"] = True
        hash2 = hash(obj)

        assert hash1 == hash2, f"Hash changed: {hash1} != {hash2}"
    finally:
        OPTIONS["use_new_combine_kwarg_defaults"] = original_option

try:
    test_hash_stability_across_options_change()
    print("Hypothesis test passed (BUG NOT REPRODUCED)")
except AssertionError as e:
    print(f"Hypothesis test failed (BUG REPRODUCED): {e}")

# Let's also check what _value returns under different options
print("\n\nChecking _value property under different options...")
obj = CombineKwargDefault(name="test", old="old_value", new="new_value")
OPTIONS["use_new_combine_kwarg_defaults"] = False
print(f"_value with option=False: {obj._value!r}")
OPTIONS["use_new_combine_kwarg_defaults"] = True
print(f"_value with option=True: {obj._value!r}")
OPTIONS["use_new_combine_kwarg_defaults"] = original_option

# Check if the object is being used as intended
print("\n\nChecking equality behavior...")
obj1 = CombineKwargDefault(name="test", old="old_value", new="new_value")
obj2 = CombineKwargDefault(name="test", old="old_value", new="new_value")
OPTIONS["use_new_combine_kwarg_defaults"] = False
eq1 = obj1 == obj2
print(f"obj1 == obj2 with option=False: {eq1}")
OPTIONS["use_new_combine_kwarg_defaults"] = True
eq2 = obj1 == obj2
print(f"obj1 == obj2 with option=True: {eq2}")
OPTIONS["use_new_combine_kwarg_defaults"] = original_option