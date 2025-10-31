#!/usr/bin/env python3
"""Test the reported bug with is_valid_nc3_name and empty string."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.backends.netcdf3 import is_valid_nc3_name

# Test 1: Try the exact reproduction case
print("Test 1: Calling is_valid_nc3_name with empty string")
try:
    result = is_valid_nc3_name("")
    print(f"Result: {result}")
    print(f"Type: {type(result)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 2: Try with various edge cases
print("\nTest 2: Edge cases")
test_cases = [
    "",           # empty string
    " ",          # single space
    "  ",         # multiple spaces
    "a",          # single valid char
    "_",          # underscore (valid first char)
    "a ",         # trailing space
    " a",         # leading space
    "valid_name", # normal valid name
]

for test_str in test_cases:
    try:
        result = is_valid_nc3_name(test_str)
        print(f"is_valid_nc3_name({repr(test_str)}): {result}")
    except Exception as e:
        print(f"is_valid_nc3_name({repr(test_str)}): ERROR - {type(e).__name__}: {e}")