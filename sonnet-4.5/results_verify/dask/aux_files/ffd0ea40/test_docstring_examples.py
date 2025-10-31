#!/usr/bin/env python3
"""Test examples from the docstring"""

from dask.dataframe.io.io import sorted_division_locations

print("Examples from docstring:")
print()

print(">>> L = ['A', 'B', 'C', 'D', 'E', 'F']")
L = ['A', 'B', 'C', 'D', 'E', 'F']
print(">>> sorted_division_locations(L, chunksize=2)")
result = sorted_division_locations(L, chunksize=2)
print(f"Expected: (['A', 'C', 'E', 'F'], [0, 2, 4, 6])")
print(f"Actual:   {result}")
print()

print(">>> sorted_division_locations(L, chunksize=3)")
result = sorted_division_locations(L, chunksize=3)
print(f"Expected: (['A', 'D', 'F'], [0, 3, 6])")
print(f"Actual:   {result}")
print()

print(">>> L = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C']")
L = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C']
print(">>> sorted_division_locations(L, chunksize=3)")
result = sorted_division_locations(L, chunksize=3)
print(f"Expected: (['A', 'B', 'C', 'C'], [0, 4, 7, 8])")
print(f"Actual:   {result}")
print()

print(">>> sorted_division_locations(L, chunksize=2)")
result = sorted_division_locations(L, chunksize=2)
print(f"Expected: (['A', 'B', 'C', 'C'], [0, 4, 7, 8])")
print(f"Actual:   {result}")
print()

print(">>> sorted_division_locations(['A'], chunksize=2)")
result = sorted_division_locations(['A'], chunksize=2)
print(f"Expected: (['A', 'A'], [0, 1])")
print(f"Actual:   {result}")
print()

# Test with npartitions instead of chunksize
print("\nTesting with npartitions:")
print()

L = ['A', 'B', 'C', 'D', 'E', 'F']
print(f"L = {L}")
print("sorted_division_locations(L, npartitions=2)")
result = sorted_division_locations(L, npartitions=2)
print(f"Result: {result}")
print(f"Number of divisions: {len(result[0])}, Expected: {2 + 1}")
print()

print("sorted_division_locations(L, npartitions=3)")
result = sorted_division_locations(L, npartitions=3)
print(f"Result: {result}")
print(f"Number of divisions: {len(result[0])}, Expected: {3 + 1}")