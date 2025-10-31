#!/usr/bin/env python3
"""Test script to reproduce the CombineKwargDefault hash mutability bug"""

from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

print("=" * 60)
print("Testing with specific failing input from report:")
print("=" * 60)

# Test with the specific failing input mentioned in the report
name, old, new = '0', '', '0'
obj = CombineKwargDefault(name=name, old=old, new=new)

with set_options(use_new_combine_kwarg_defaults=False):
    hash1 = hash(obj)
    print(f"Hash with use_new_combine_kwarg_defaults=False: {hash1}")
    print(f"  _value: '{obj._value}'")

with set_options(use_new_combine_kwarg_defaults=True):
    hash2 = hash(obj)
    print(f"Hash with use_new_combine_kwarg_defaults=True: {hash2}")
    print(f"  _value: '{obj._value}'")

print(f"\nHashes are equal? {hash1 == hash2}")
if hash1 != hash2:
    print(f"ERROR: Hash changed from {hash1} to {hash2}")

print("\n" + "=" * 60)
print("Running reproduction example from bug report:")
print("=" * 60)

obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

with set_options(use_new_combine_kwarg_defaults=False):
    hash1 = hash(obj)
    s = {obj}
    assert obj in s
    print(f"With use_new_combine_kwarg_defaults=False:")
    print(f"  Hash: {hash1}")
    print(f"  Object._value: {obj._value}")
    print(f"  Object in set: {obj in s}")

with set_options(use_new_combine_kwarg_defaults=True):
    hash2 = hash(obj)
    print(f"\nWith use_new_combine_kwarg_defaults=True:")
    print(f"  Hash: {hash2}")
    print(f"  Object._value: {obj._value}")
    print(f"  Object in set: {obj in s}")
    print(f"\nHash changed: {hash1} -> {hash2}")
    print(f"Same hash? {hash1 == hash2}")

print("\n" + "=" * 60)
print("Testing dict/set behavior:")
print("=" * 60)

# Test with dictionary
obj2 = CombineKwargDefault(name="dict_test", old="old", new="new")

with set_options(use_new_combine_kwarg_defaults=False):
    d = {obj2: "value_in_dict"}
    print(f"Added to dict with options=False, hash: {hash(obj2)}")
    print(f"Can retrieve from dict: {obj2 in d}")

with set_options(use_new_combine_kwarg_defaults=True):
    print(f"Hash after setting options=True: {hash(obj2)}")
    print(f"Can still retrieve from dict: {obj2 in d}")
    try:
        retrieved = d[obj2]
        print(f"Retrieved value: {retrieved}")
    except KeyError:
        print("ERROR: KeyError - Object cannot be found in dict after hash changed!")

# Test the problematic set behavior
print("\n" + "=" * 60)
print("Testing set membership after option change:")
print("=" * 60)

obj3 = CombineKwargDefault(name="set_test", old="val1", new="val2")

with set_options(use_new_combine_kwarg_defaults=False):
    test_set = {obj3}
    hash_before = hash(obj3)
    print(f"Object added to set with hash: {hash_before}")
    print(f"Object in set: {obj3 in test_set}")

with set_options(use_new_combine_kwarg_defaults=True):
    hash_after = hash(obj3)
    print(f"Hash after option change: {hash_after}")
    print(f"Object still in set: {obj3 in test_set}")
    if obj3 not in test_set:
        print("ERROR: Object lost from set due to hash change!")