#!/usr/bin/env python3
"""Debug the exact line where the error occurs."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

# Test accessing s[-1] on empty string directly
s = ""
print(f"Testing s[-1] on empty string:")
print(f"s = {repr(s)}")
print(f"len(s) = {len(s)}")

try:
    result = s[-1]
    print(f"s[-1] = {repr(result)}")
except IndexError as e:
    print(f"Error: {e}")

# Test on s[0] too
print(f"\nTesting s[0] on empty string:")
try:
    result = s[0]
    print(f"s[0] = {repr(result)}")
except IndexError as e:
    print(f"Error: {e}")

# Test the alternative endswith approach
print(f"\nTesting s.endswith(' ') on empty string:")
result = s.endswith(" ")
print(f"s.endswith(' ') = {result}")

print(f"\nTesting not s.endswith(' ') on empty string:")
result = not s.endswith(" ")
print(f"not s.endswith(' ') = {result}")