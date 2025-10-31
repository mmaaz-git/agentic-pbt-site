#!/usr/bin/env python3
"""Test the exact bug case from the report."""

from pandas.arrays import SparseArray

print("=== Exact Bug Report Case ===")
arr = SparseArray([0, 0, 1, 2], fill_value=0)
print(f"Original array: {arr.to_dense()}")
print(f"Original density: {arr.density}")
print(f"Original sparse values: {arr.sp_values}")

# The exact mapper from the bug report
mapper = lambda x: 0 if x == 1 else x

print(f"\nMapping with: lambda x: 0 if x == 1 else x")
print("This would map value 1 -> 0 (the fill value)")
print("Expected behavior according to docstring: preserve density")

try:
    mapped = arr.map(mapper)
    print(f"✗ UNEXPECTED: Mapping succeeded!")
    print(f"Mapped array: {mapped.to_dense()}")
    print(f"Mapped density: {mapped.density}")
    print(f"Mapped sparse values: {mapped.sp_values}")
except ValueError as e:
    print(f"✓ Got expected ValueError: {e}")

print("\n=== Understanding the Issue ===")
print("The bug report states that the docstring claims:")
print('"The output array will have the same density as the input"')
print("\nBut when mapping a sparse value (1) to the fill value (0),")
print("the function raises a ValueError instead of handling it.")
print("\nThis is a contradiction:")
print("- The docstring promises density preservation")
print("- But the implementation rejects mappings that would require recalculation")