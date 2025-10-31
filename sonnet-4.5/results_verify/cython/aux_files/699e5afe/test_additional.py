#!/usr/bin/env python3
"""Additional tests to understand the behavior"""

from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

test_cases = [
    ".0",
    ".123",
    ".999",
    ".1a",  # Not all decimal
    ".",     # Just a dot
    ".a",    # Not decimal
    "0.0",   # Doesn't start with dot
    "",      # Empty string
]

print("Testing regular strings:")
for test in test_cases:
    result = is_valid_tag(test)
    print(f"  is_valid_tag('{test}') = {result}")

print("\nTesting EncodedString instances:")
for test in test_cases:
    encoded = EncodedString(test)
    result = is_valid_tag(encoded)
    print(f"  is_valid_tag(EncodedString('{test}')) = {result}")