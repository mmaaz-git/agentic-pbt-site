#!/usr/bin/env python3
"""Minimal reproduction of the unquote function ValueError bug."""

from dask.diagnostics.profile_visualize import unquote

# Test case 1: Empty list in dict task
print("Test 1: (dict, [[]])")
try:
    task = (dict, [[]])
    result = unquote(task)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print()

# Test case 2: Empty list as argument to dict
print("Test 2: (dict, [])")
try:
    task = (dict, [])
    result = unquote(task)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")