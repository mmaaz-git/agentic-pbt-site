#!/usr/bin/env python3
"""Test for reproducing the SparseArray.map() density preservation bug."""

from pandas.arrays import SparseArray

# First, let's run the simple reproduction case
print("=== Simple Reproduction Case ===")
arr = SparseArray([0, 0, 1, 2], fill_value=0)
print(f"Original array: {arr.to_dense()}")
print(f"Original density: {arr.density}")
print(f"Sparse values: {arr.sp_values}")
print(f"Fill value: {arr.fill_value}")

try:
    mapped = arr.map(lambda x: 0 if x == 1 else x)
    print(f"Mapped density: {mapped.density}")
except ValueError as e:
    print(f"Error: {e}")

print("\n=== Manual test cases ===")

def manual_test_map_density(data, fill_value):
    arr = SparseArray(data, fill_value=fill_value)

    if arr.npoints == 0:
        print(f"Skipped (no sparse values): data={data}, fill_value={fill_value}")
        return

    sparse_val = arr.sp_values[0]
    mapper = lambda x: fill_value if x == sparse_val else x + 100

    print(f"\nTesting data={data}, fill_value={fill_value}")
    print(f"  Original density: {arr.density}")
    print(f"  Sparse values: {arr.sp_values}")
    print(f"  Mapping first sparse value {sparse_val} to fill value {fill_value}")

    try:
        mapped = arr.map(mapper)
        print(f"  Mapped density: {mapped.density}")
        if mapped.density == arr.density:
            print(f"  ✓ Density preserved")
        else:
            print(f"  ✗ Density changed from {arr.density} to {mapped.density}")
    except ValueError as e:
        print(f"  ✗ Error raised: {e}")

# Run test cases
test_cases = [
    ([0, 0, 1, 2], 0),  # The reported failing case
    ([1, 1, 2, 3], 1),
    ([5, 5, 5, 6], 5),
    ([0, 1, 1, 1], 0),  # Multiple sparse values
]

for data, fill_value in test_cases:
    manual_test_map_density(data, fill_value)

print("\n=== Additional test: mapping that doesn't hit fill value ===")
arr2 = SparseArray([0, 0, 1, 2], fill_value=0)
print(f"Original density: {arr2.density}")
try:
    # This should work - mapping to values that aren't the fill value
    mapped2 = arr2.map(lambda x: x + 10)
    print(f"Mapped density: {mapped2.density}")
    print(f"Mapped values: {mapped2.to_dense()}")
    print("✓ Mapping succeeded when not mapping to fill value")
except Exception as e:
    print(f"✗ Unexpected error: {e}")