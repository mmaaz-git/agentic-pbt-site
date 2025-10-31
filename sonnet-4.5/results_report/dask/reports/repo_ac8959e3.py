#!/usr/bin/env python3
"""Minimal reproduction case for dask.diagnostics.profile_visualize.unquote bug"""

from dask.diagnostics.profile_visualize import unquote

print("Test 1: unquote((dict, []))")
print("=" * 50)
try:
    result = unquote((dict, []))
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTest 2: unquote((dict, [[]]))")
print("=" * 50)
try:
    result = unquote((dict, [[]]))
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")