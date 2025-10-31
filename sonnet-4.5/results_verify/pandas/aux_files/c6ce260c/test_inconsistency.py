#!/usr/bin/env python3
"""Demonstrate the inconsistency in check_array_indexer for empty arrays."""

import numpy as np
import pandas as pd
from pandas.api.indexers import check_array_indexer

print("=== Demonstrating the inconsistency ===\n")

# Use a non-empty array to avoid the boolean length check issue
array = pd.array([1, 2, 3])
print(f"Target array: {array}\n")

print("1. Empty Python list as indexer:")
empty_list = []
print(f"   Type: {type(empty_list)}")
try:
    result = check_array_indexer(array, empty_list)
    print(f"   ✓ Result: {result} (dtype: {result.dtype})")
    print(f"   ✓ Successfully returns empty integer array")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n2. Empty pandas array (default dtype) as indexer:")
empty_pd_default = pd.array([])
print(f"   Type: {type(empty_pd_default)}")
print(f"   Dtype: {empty_pd_default.dtype}")
try:
    result = check_array_indexer(array, empty_pd_default)
    print(f"   ✓ Result: {result} (dtype: {result.dtype})")
    print(f"   ✓ Successfully returns empty array")
except Exception as e:
    print(f"   ✗ Error: {e}")
    print(f"   ✗ Fails even though it's logically equivalent to empty list")

print("\n3. Empty pandas array (int dtype) as indexer:")
empty_pd_int = pd.array([], dtype='int64')
print(f"   Type: {type(empty_pd_int)}")
print(f"   Dtype: {empty_pd_int.dtype}")
try:
    result = check_array_indexer(array, empty_pd_int)
    print(f"   ✓ Result: {result} (dtype: {result.dtype})")
    print(f"   ✓ Successfully returns empty integer array")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n=== Analysis ===")
print("The function treats logically equivalent empty inputs differently:")
print("- Empty Python list [] → Succeeds (converted to np.array([], dtype=np.intp))")
print("- Empty pandas array pd.array([]) → Fails (has Float64 dtype)")
print("- Empty pandas array pd.array([], dtype='int64') → Succeeds")
print("\nThis is inconsistent behavior for functionally equivalent empty indexers.")

# Show what's happening in the code
print("\n=== Code Path Analysis ===")
from pandas.core.dtypes.common import is_array_like

empty_list = []
empty_pd = pd.array([])

print(f"empty_list is_array_like: {is_array_like(empty_list)}")
print(f"empty_pd is_array_like: {is_array_like(empty_pd)}")
print("\nThe empty list is NOT array-like, so it gets special handling (lines 526-528)")
print("The empty pandas array IS array-like, so it skips the special handling")
print("and fails the dtype check at line 551 because it has Float64 dtype")