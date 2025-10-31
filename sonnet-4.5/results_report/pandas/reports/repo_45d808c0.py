#!/usr/bin/env python3
"""Minimal reproduction of the cap_length bug."""

from Cython.Compiler import PyrexTypes

# Test case from the bug report
s = '00000000000'
max_len = 10

result = PyrexTypes.cap_length(s, max_len)

print(f"Input string: {s!r}")
print(f"Input length: {len(s)}")
print(f"Max length: {max_len}")
print(f"Result: {result!r}")
print(f"Result length: {len(result)}")
print(f"Expected: <= {max_len}")
print(f"Bug: {len(result)} > {max_len} (should be at most {max_len})")