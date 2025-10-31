#!/usr/bin/env python3
"""Minimal reproduction of the dask.utils.key_split bug."""

from dask.utils import key_split

# Test 1: Invalid UTF-8 bytes
print("Test 1: Invalid UTF-8 bytes")
try:
    result = key_split(b'\x80')
    print(f"key_split(b'\\x80') = {result}")
except Exception as e:
    print(f"BUG: Raised {type(e).__name__}: {e}")

# Test 2: Unhashable type - list
print("\nTest 2: Unhashable type - list")
try:
    result = key_split([])
    print(f"key_split([]) = {result}")
except Exception as e:
    print(f"BUG: Raised {type(e).__name__}: {e}")

# Test 3: Unhashable type - dict
print("\nTest 3: Unhashable type - dict")
try:
    result = key_split({})
    print(f"key_split({{}}) = {result}")
except Exception as e:
    print(f"BUG: Raised {type(e).__name__}: {e}")

# Test 4: None (should work)
print("\nTest 4: None (should return 'Other')")
try:
    result = key_split(None)
    print(f"key_split(None) = {result}")
except Exception as e:
    print(f"BUG: Raised {type(e).__name__}: {e}")

# Test 5: Valid string (should work)
print("\nTest 5: Valid string")
try:
    result = key_split('x-1-2-3')
    print(f"key_split('x-1-2-3') = {result}")
except Exception as e:
    print(f"BUG: Raised {type(e).__name__}: {e}")

# Test 6: Valid bytes (should work)
print("\nTest 6: Valid UTF-8 bytes")
try:
    result = key_split(b'hello-world')
    print(f"key_split(b'hello-world') = {result}")
except Exception as e:
    print(f"BUG: Raised {type(e).__name__}: {e}")