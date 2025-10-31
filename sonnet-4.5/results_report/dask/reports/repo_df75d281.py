"""Minimal reproduction of sorted_division_locations bug with plain lists."""
from dask.dataframe.io.io import sorted_division_locations

# Example 1: Simple list from docstring
L = ['A', 'B', 'C', 'D', 'E', 'F']
print("Testing Example 1: L = ['A', 'B', 'C', 'D', 'E', 'F'] with chunksize=2")
try:
    result = sorted_division_locations(L, chunksize=2)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting Example 2: L = ['A', 'B', 'C', 'D', 'E', 'F'] with chunksize=3")
try:
    result = sorted_division_locations(L, chunksize=3)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Example 3: List with duplicates from docstring
L2 = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C']
print("\nTesting Example 3: L = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C'] with chunksize=3")
try:
    result = sorted_division_locations(L2, chunksize=3)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Minimal example
print("\nTesting Minimal Example: L = ['A'] with chunksize=2")
L3 = ['A']
try:
    result = sorted_division_locations(L3, chunksize=2)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")