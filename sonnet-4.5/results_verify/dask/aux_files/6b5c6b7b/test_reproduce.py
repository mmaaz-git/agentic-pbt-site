#!/usr/bin/env python3
from dask.utils import key_split

print("Testing with valid UTF-8 bytes:")
result = key_split(b'hello-world-1')
print(f"key_split(b'hello-world-1') = '{result}'")

print("\nTesting with invalid UTF-8 bytes:")
try:
    result = key_split(b'\x80')
    print(f"key_split(b'\\x80') = '{result}'")
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}")