#!/usr/bin/env python3
"""Test the reported SparseArray fill_value setter bug."""

import numpy as np
from pandas.arrays import SparseArray

# Test 1: Simple reproduction case
print("=" * 60)
print("Test 1: Simple reproduction case")
print("=" * 60)

data = [0, 0, 0]
sparse = SparseArray(data, fill_value=0)
print(f"Original data: {sparse.to_dense()}")
print(f"Original fill_value: {sparse.fill_value}")

sparse.fill_value = 999
print(f"After setting fill_value=999: {sparse.to_dense()}")
print(f"New fill_value: {sparse.fill_value}")

# Test 2: When data contains non-fill values
print("\n" + "=" * 60)
print("Test 2: When data contains non-fill values")
print("=" * 60)

data2 = [0, 1, 0, 2, 0]
sparse2 = SparseArray(data2, fill_value=0)
print(f"Original data: {sparse2.to_dense()}")
print(f"Original fill_value: {sparse2.fill_value}")

sparse2.fill_value = 999
print(f"After setting fill_value=999: {sparse2.to_dense()}")
print(f"New fill_value: {sparse2.fill_value}")

# Test 3: Property-based test with specific failing input
print("\n" + "=" * 60)
print("Test 3: Property-based test with failing input [0]")
print("=" * 60)

data3 = [0]
sparse3 = SparseArray(data3, fill_value=0)
original_dense = sparse3.to_dense()
print(f"Original data: {original_dense}")

sparse3.fill_value = 999
new_dense = sparse3.to_dense()
print(f"After setting fill_value=999: {new_dense}")
print(f"Are they equal? {np.array_equal(original_dense, new_dense)}")

# Test 4: Understanding the internal representation
print("\n" + "=" * 60)
print("Test 4: Internal representation")
print("=" * 60)

data4 = [0, 0, 1, 0, 2]
sparse4 = SparseArray(data4, fill_value=0)
print(f"Data: {data4}")
print(f"fill_value: {sparse4.fill_value}")
print(f"sp_values (non-fill values): {sparse4.sp_values}")
print(f"sp_index.indices: {sparse4.sp_index.indices}")
print(f"Dense representation: {sparse4.to_dense()}")

print("\nChanging fill_value to 999...")
sparse4.fill_value = 999
print(f"New fill_value: {sparse4.fill_value}")
print(f"sp_values (unchanged): {sparse4.sp_values}")
print(f"sp_index.indices (unchanged): {sparse4.sp_index.indices}")
print(f"Dense representation: {sparse4.to_dense()}")

# Test 5: Edge case - no values equal to fill_value
print("\n" + "=" * 60)
print("Test 5: No values equal to original fill_value")
print("=" * 60)

data5 = [1, 2, 3, 4, 5]
sparse5 = SparseArray(data5, fill_value=0)
original5 = sparse5.to_dense()
print(f"Original data: {original5}")

sparse5.fill_value = 999
new5 = sparse5.to_dense()
print(f"After setting fill_value=999: {new5}")
print(f"Are they equal? {np.array_equal(original5, new5)}")