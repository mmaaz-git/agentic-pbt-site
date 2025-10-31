#!/usr/bin/env python3
"""Minimal reproduction of Cython Shadow index_type bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Shadow import index_type

print("Testing Cython Shadow index_type step validation bug")
print("=" * 60)

# Test case 1: 2D array with step in both dimensions (should fail but doesn't)
print("\nTest 1: 2D array with step in both dimensions")
print("slices = (slice(None, None, 1), slice(None, None, 1))")
try:
    slices_2d = (slice(None, None, 1), slice(None, None, 1))
    result = index_type(int, slices_2d)
    print(f"Result: {result}")
    print("ERROR: Should have raised InvalidTypeSpecification!")
except Exception as e:
    print(f"Raised exception: {e}")

# Test case 2: 3D array with step in first and last dimensions (should fail but doesn't)
print("\nTest 2: 3D array with step in first and last dimensions")
print("slices = (slice(None, None, 1), slice(None, None, None), slice(None, None, 1))")
try:
    slices_3d = (slice(None, None, 1), slice(None, None, None), slice(None, None, 1))
    result = index_type(int, slices_3d)
    print(f"Result: {result}")
    print("ERROR: Should have raised InvalidTypeSpecification!")
except Exception as e:
    print(f"Raised exception: {e}")

# Test case 3: Valid - 2D array with step only in last dimension (should succeed)
print("\nTest 3: Valid - 2D array with step only in last dimension")
print("slices = (slice(None, None, None), slice(None, None, 1))")
try:
    slices_valid = (slice(None, None, None), slice(None, None, 1))
    result = index_type(int, slices_valid)
    print(f"Result: {result}")
    print("Success: This is valid and works correctly")
except Exception as e:
    print(f"Raised exception: {e}")

# Test case 4: Valid - 2D array with step only in first dimension (should succeed)
print("\nTest 4: Valid - 2D array with step only in first dimension")
print("slices = (slice(None, None, 1), slice(None, None, None))")
try:
    slices_valid2 = (slice(None, None, 1), slice(None, None, None))
    result = index_type(int, slices_valid2)
    print(f"Result: {result}")
    print("Success: This is valid and works correctly")
except Exception as e:
    print(f"Raised exception: {e}")

# Test case 5: Invalid - 3D array with step in middle dimension (should fail and does)
print("\nTest 5: Invalid - 3D array with step in middle dimension")
print("slices = (slice(None, None, None), slice(None, None, 1), slice(None, None, None))")
try:
    slices_invalid = (slice(None, None, None), slice(None, None, 1), slice(None, None, None))
    result = index_type(int, slices_invalid)
    print(f"Result: {result}")
    print("ERROR: Should have raised InvalidTypeSpecification!")
except Exception as e:
    print(f"Raised exception: {e}")
    print("Success: This correctly raises an error")