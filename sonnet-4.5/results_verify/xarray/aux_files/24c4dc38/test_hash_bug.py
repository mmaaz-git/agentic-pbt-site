#!/usr/bin/env python3
"""Test to reproduce the CombineKwargDefault hash stability bug."""

import xarray
from xarray.util.deprecation_helpers import CombineKwargDefault

print("="*60)
print("Test 1: Basic hash stability test")
print("="*60)

obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

xarray.set_options(use_new_combine_kwarg_defaults=False)
hash1 = hash(obj)
print(f"Hash with old defaults: {hash1}")

xarray.set_options(use_new_combine_kwarg_defaults=True)
hash2 = hash(obj)
print(f"Hash with new defaults: {hash2}")

if hash1 == hash2:
    print("✓ Hashes are equal (expected behavior)")
else:
    print(f"✗ Hash changed: {hash1} != {hash2} (BUG)")

print("\n" + "="*60)
print("Test 2: Dict lookup failure")
print("="*60)

xarray.set_options(use_new_combine_kwarg_defaults=False)
obj = CombineKwargDefault(name="param", old="old", new="new")
test_dict = {obj: "value"}
print(f"Object inserted into dict with OPTIONS['use_new_combine_kwarg_defaults'] = False")

# Check lookup with same settings
result1 = obj in test_dict
print(f"Lookup with same settings: {result1}")

xarray.set_options(use_new_combine_kwarg_defaults=True)
result2 = obj in test_dict
print(f"Lookup after changing OPTIONS to True: {result2}")

if not result2:
    print("✗ Object cannot be found in dict after OPTIONS change (BUG)")

print("\n" + "="*60)
print("Test 3: Set membership failure")
print("="*60)

xarray.set_options(use_new_combine_kwarg_defaults=False)
obj = CombineKwargDefault(name="param", old="old", new="new")
test_set = {obj}
print(f"Object added to set with OPTIONS['use_new_combine_kwarg_defaults'] = False")

result1 = obj in test_set
print(f"Membership test with same settings: {result1}")

xarray.set_options(use_new_combine_kwarg_defaults=True)
result2 = obj in test_set
print(f"Membership test after changing OPTIONS to True: {result2}")

if not result2:
    print("✗ Object not found in set after OPTIONS change (BUG)")

print("\n" + "="*60)
print("Test 4: Hypothesis property test")
print("="*60)

from hypothesis import given, strategies as st

@given(
    name=st.text(min_size=1, max_size=20),
    old_val=st.text(min_size=1, max_size=10),
    new_val=st.text(min_size=1, max_size=10) | st.none()
)
def test_hash_stability(name, old_val, new_val):
    """Hash of an object should remain constant during its lifetime."""
    obj = CombineKwargDefault(name=name, old=old_val, new=new_val)

    xarray.set_options(use_new_combine_kwarg_defaults=False)
    hash1 = hash(obj)

    xarray.set_options(use_new_combine_kwarg_defaults=True)
    hash2 = hash(obj)

    assert hash1 == hash2, f"Hash changed from {hash1} to {hash2} when options changed"

try:
    test_hash_stability()
    print("✓ Hypothesis test passed (no failures found)")
except AssertionError as e:
    print(f"✗ Hypothesis test failed: {e}")

print("\n" + "="*60)
print("Test 5: Check equality behavior")
print("="*60)

xarray.set_options(use_new_combine_kwarg_defaults=False)
obj1 = CombineKwargDefault(name="test", old="old", new="new")
obj2 = CombineKwargDefault(name="test", old="old", new="new")

print(f"obj1 == obj2 with old defaults: {obj1 == obj2}")
print(f"hash(obj1) == hash(obj2) with old defaults: {hash(obj1) == hash(obj2)}")

xarray.set_options(use_new_combine_kwarg_defaults=True)
print(f"obj1 == obj2 with new defaults: {obj1 == obj2}")
print(f"hash(obj1) == hash(obj2) with new defaults: {hash(obj1) == hash(obj2)}")

print("\n" + "="*60)
print("Test 6: Check what _value property returns")
print("="*60)

obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

xarray.set_options(use_new_combine_kwarg_defaults=False)
print(f"_value with old defaults: {obj._value!r}")

xarray.set_options(use_new_combine_kwarg_defaults=True)
print(f"_value with new defaults: {obj._value!r}")

print("\n" + "="*60)
print("Summary")
print("="*60)
print("The CombineKwargDefault class has a hash that changes based on global OPTIONS state.")
print("This violates Python's hash contract and breaks dict/set lookups.")