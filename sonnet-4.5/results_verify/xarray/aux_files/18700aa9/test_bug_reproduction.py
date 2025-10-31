#!/usr/bin/env python3
"""Test to reproduce the CombineKwargDefault.__dask_tokenize__ bug."""

from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

print("Testing CombineKwargDefault.__dask_tokenize__ mutability...")
print("=" * 60)

obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

with set_options(use_new_combine_kwarg_defaults=False):
    token1 = obj.__dask_tokenize__()
    print(f"Token with use_new=False: {token1}")

with set_options(use_new_combine_kwarg_defaults=True):
    token2 = obj.__dask_tokenize__()
    print(f"Token with use_new=True: {token2}")

print(f"\nTokens are equal: {token1 == token2}")

if token1 != token2:
    print("\n❌ BUG CONFIRMED: The dask token changes based on global OPTIONS!")
    print("   This violates dask's requirement that an object's token")
    print("   must remain constant for proper caching and memoization.")
else:
    print("\n✓ No bug: Tokens are identical")