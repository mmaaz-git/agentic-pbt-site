#!/usr/bin/env python3
"""Verify Python's standard list.insert behavior."""

# Test Python's list.insert
lst = ['a', 'b', 'c']
print(f"Original list: {lst}")
print(f"Original length: {len(lst)}")

lst.insert(1, 'X')
print(f"\nAfter lst.insert(1, 'X'):")
print(f"New list: {lst}")
print(f"New length: {len(lst)}")
print(f"Length increased: {len(['a', 'b', 'c']) < len(lst)}")

# Test at position 0
lst2 = ['a', 'b', 'c']
lst2.insert(0, 'x')
print(f"\nlst.insert(0, 'x') on ['a', 'b', 'c']: {lst2}")

# Compare with dask.utils.insert
from dask.utils import insert

tup = ('a', 'b', 'c')
result = insert(tup, 0, 'x')
print(f"\ndask.utils.insert(('a', 'b', 'c'), 0, 'x'): {result}")

print("\n=== Key Difference ===")
print(f"Python list.insert: ADDS element, increases length")
print(f"dask.utils.insert: REPLACES element, preserves length")