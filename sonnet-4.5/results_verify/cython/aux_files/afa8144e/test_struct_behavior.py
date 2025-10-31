#!/usr/bin/env python3
"""Test struct.unpack behavior with mismatched buffer sizes"""

import struct

print("Testing struct.unpack behavior:")
print("=" * 60)

# Test case 1: Exact match (should work)
data1 = b'\x00\x00\x80\x3f\x00\x00\x00\x40\x00\x00\x40\x40'  # 12 bytes
format1 = "<fff"  # expects 12 bytes
print(f"Test 1: 12 bytes buffer, format '<fff' (expects 12 bytes)")
try:
    result = struct.unpack(format1, data1)
    print(f"  Success: {result}")
except struct.error as e:
    print(f"  Error: {e}")

# Test case 2: Buffer too small (should fail)
data2 = b'\x00\x00\x80\x3f\x00\x00\x00\x40'  # 8 bytes
format2 = "<fff"  # expects 12 bytes
print(f"\nTest 2: 8 bytes buffer, format '<fff' (expects 12 bytes)")
try:
    result = struct.unpack(format2, data2)
    print(f"  Success: {result}")
except struct.error as e:
    print(f"  Error: {e}")

# Test case 3: Buffer too large (should fail in Python 3)
data3 = b'\x00\x00\x80\x3f\x00\x00\x00\x40\x00\x00\x40\x40\xFF'  # 13 bytes
format3 = "<fff"  # expects 12 bytes
print(f"\nTest 3: 13 bytes buffer, format '<fff' (expects 12 bytes)")
try:
    result = struct.unpack(format3, data3)
    print(f"  Success: {result}")
    print(f"  WARNING: Silent truncation occurred!")
except struct.error as e:
    print(f"  Error: {e}")

# Test case 4: Using unpack_from (which can handle larger buffers)
print(f"\nTest 4: Using unpack_from with 13 bytes buffer")
try:
    result = struct.unpack_from(format3, data3, 0)
    print(f"  Success with unpack_from: {result}")
    print(f"  This reads only the first 12 bytes, ignoring the rest")
except struct.error as e:
    print(f"  Error: {e}")

# Let's test if there's any Python version where this might work differently
print("\n" + "=" * 60)
print("Testing if buffer slicing makes a difference:")

data5 = b'\x00\x00\x80\x3f\x00\x00\x00\x40\x00\x00\x40\x40\xFF\xFF\xFF\xFF'  # 16 bytes
print(f"Data: {len(data5)} bytes")

# Manual slicing
format5 = "<" + "f" * (len(data5) // 4)
sliced_data = data5[:struct.calcsize(format5)]
print(f"Format: '{format5}' (expects {struct.calcsize(format5)} bytes)")
print(f"Sliced data: {len(sliced_data)} bytes")
result = struct.unpack(format5, sliced_data)
print(f"Result after manual slicing: {result}")
print("This is what would happen if decode() sliced the buffer before unpacking")