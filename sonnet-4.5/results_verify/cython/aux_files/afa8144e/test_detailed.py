#!/usr/bin/env python3
"""Detailed test of the decode function behavior"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm
import struct

# Test the exact scenario from the bug report
print("Testing the exact reproduction case from bug report:")
print("=" * 60)

values = [1.0, 2.0, 3.0]
encoded = llm.encode(values)
print(f"Original values: {values}")
print(f"Encoded bytes: {encoded}")
print(f"Encoded length: {len(encoded)} bytes")

# Add one extra byte to simulate corruption
corrupted_bytes = encoded + b'\x00'
print(f"\nCorrupted bytes (added 1 byte): {corrupted_bytes}")
print(f"Corrupted length: {len(corrupted_bytes)} bytes (not multiple of 4)")

# Try to decode the corrupted bytes
try:
    decoded = llm.decode(corrupted_bytes)
    print(f"Decoded successfully: {list(decoded)}")
    print(f"Lost data: {len(corrupted_bytes) - len(decoded) * 4} bytes silently discarded")
    print("BUG CONFIRMED: Silent truncation occurred!")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Testing the decode implementation manually:")
print("=" * 60)

# Let's manually test what the function does
binary = corrupted_bytes
format_string = "<" + "f" * (len(binary) // 4)
print(f"Binary length: {len(binary)}")
print(f"len(binary) // 4 = {len(binary) // 4}")
print(f"Format string: '{format_string}'")
print(f"Expected bytes for format: {struct.calcsize(format_string)}")

# Test what struct.unpack actually does
try:
    result = struct.unpack(format_string, binary)
    print(f"struct.unpack result: {result}")
    print("BUG: struct.unpack succeeded with mismatched buffer size!")
except struct.error as e:
    print(f"struct.error raised: {e}")

# Let's verify the actual behavior more carefully
print("\n" + "=" * 60)
print("Additional tests:")
print("=" * 60)

# Test with 12 bytes (3 floats)
test1 = b'\x00\x00\x80\x3f\x00\x00\x00\x40\x00\x00\x40\x40'  # [1.0, 2.0, 3.0]
print(f"Test 1 - 12 bytes (valid): {llm.decode(test1)}")

# Test with 13 bytes (3 floats + 1 byte)
test2 = b'\x00\x00\x80\x3f\x00\x00\x00\x40\x00\x00\x40\x40\xFF'
print(f"Test 2 - 13 bytes (invalid):")
try:
    result = llm.decode(test2)
    print(f"  Result: {result}")
    print(f"  Length: {len(result)} floats")
except Exception as e:
    print(f"  Exception: {e}")

# Test with 14 bytes (3 floats + 2 bytes)
test3 = b'\x00\x00\x80\x3f\x00\x00\x00\x40\x00\x00\x40\x40\xFF\xFF'
print(f"Test 3 - 14 bytes (invalid):")
try:
    result = llm.decode(test3)
    print(f"  Result: {result}")
    print(f"  Length: {len(result)} floats")
except Exception as e:
    print(f"  Exception: {e}")

# Test with 15 bytes (3 floats + 3 bytes)
test4 = b'\x00\x00\x80\x3f\x00\x00\x00\x40\x00\x00\x40\x40\xFF\xFF\xFF'
print(f"Test 4 - 15 bytes (invalid):")
try:
    result = llm.decode(test4)
    print(f"  Result: {result}")
    print(f"  Length: {len(result)} floats")
except Exception as e:
    print(f"  Exception: {e}")