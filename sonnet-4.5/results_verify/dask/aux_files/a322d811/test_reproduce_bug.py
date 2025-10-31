#!/usr/bin/env python3
"""Test to reproduce the reported bug in dask.utils.key_split"""

from dask.utils import key_split

print("Testing key_split with invalid UTF-8 bytes:")
try:
    result = key_split(b'\x80')
    print(f"key_split(b'\\x80') = {result}")
except Exception as e:
    print(f"BUG: Raised {type(e).__name__}: {e}")

print("\nTesting key_split with unhashable type (list):")
try:
    result = key_split([])
    print(f"key_split([]) = {result}")
except Exception as e:
    print(f"BUG: Raised {type(e).__name__}: {e}")

print("\nTesting key_split with unhashable type (dict):")
try:
    result = key_split({})
    print(f"key_split({{}}) = {result}")
except Exception as e:
    print(f"BUG: Raised {type(e).__name__}: {e}")

print("\nTesting key_split with None:")
try:
    result = key_split(None)
    print(f"key_split(None) = {result}")
except Exception as e:
    print(f"BUG: Raised {type(e).__name__}: {e}")

print("\nTesting key_split with valid inputs:")
print(f"key_split('x') = {key_split('x')}")
print(f"key_split(b'hello') = {key_split(b'hello')}")
print(f"key_split(('test',)) = {key_split(('test',))}")