#!/usr/bin/env python3

# Test Python's native string handling with null bytes
s1 = "hello\x00"
s2 = "abc\x00\x00\x00"
s3 = "\x00"

print("=== Python Native String Multiplication ===")
print(f"s1 = {repr(s1)}")
print(f"s1 * 3 = {repr(s1 * 3)}")
print(f"Length of s1: {len(s1)}")
print(f"Length of s1 * 3: {len(s1 * 3)}")

print(f"\ns2 = {repr(s2)}")
print(f"s2 * 2 = {repr(s2 * 2)}")
print(f"Length of s2: {len(s2)}")
print(f"Length of s2 * 2: {len(s2 * 2)}")

print(f"\ns3 = {repr(s3)}")
print(f"s3 * 5 = {repr(s3 * 5)}")
print(f"Length of s3: {len(s3)}")
print(f"Length of s3 * 5: {len(s3 * 5)}")

# Verify null bytes are preserved
result = "test\x00" * 2
assert result == "test\x00test\x00"
assert len(result) == 10  # 4 + 1 + 4 + 1
print("\n✓ Python preserves null bytes in string multiplication")