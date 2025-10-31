#!/usr/bin/env python3
"""Test the hash behavior of CombineKwargDefault."""

from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

print("Testing CombineKwargDefault.__hash__() behavior...")
print("=" * 60)

obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

with set_options(use_new_combine_kwarg_defaults=False):
    hash1 = hash(obj)
    value1 = obj._value
    print(f"Hash with use_new=False: {hash1}")
    print(f"Value with use_new=False: {value1}")

with set_options(use_new_combine_kwarg_defaults=True):
    hash2 = hash(obj)
    value2 = obj._value
    print(f"Hash with use_new=True: {hash2}")
    print(f"Value with use_new=True: {value2}")

print(f"\nHashes are equal: {hash1 == hash2}")

if hash1 != hash2:
    print("\n❌ ISSUE: The hash changes based on global OPTIONS!")
    print("   This violates Python's requirement that hash must")
    print("   be constant for an object's lifetime.")
else:
    print("\n✓ No issue: Hashes are identical")