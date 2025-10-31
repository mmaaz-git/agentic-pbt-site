#!/usr/bin/env python3
"""Test the reported bug in dask.dataframe.utils.valid_divisions"""

from dask.dataframe.utils import valid_divisions
import traceback

print("=" * 60)
print("Testing valid_divisions with small inputs")
print("=" * 60)

# Test 1: Empty list
print("\n1. Testing valid_divisions([]):")
try:
    result = valid_divisions([])
    print(f"   Result: {result}")
    print(f"   Type: {type(result)}")
except IndexError as e:
    print(f"   ERROR: IndexError: {e}")
    print(f"   This is the bug - function crashes instead of returning False")

# Test 2: Single element list
print("\n2. Testing valid_divisions([1]):")
try:
    result = valid_divisions([1])
    print(f"   Result: {result}")
    print(f"   Type: {type(result)}")
except IndexError as e:
    print(f"   ERROR: IndexError: {e}")
    print(f"   This is the bug - function crashes instead of returning False")

# Test 3: Two element list (should work)
print("\n3. Testing valid_divisions([1, 2]):")
try:
    result = valid_divisions([1, 2])
    print(f"   Result: {result}")
    print(f"   Type: {type(result)}")
    print(f"   This works correctly")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

# Test 4: Check what the actual crash location is
print("\n4. Verifying the crash location:")
print("   Looking at line 715 in valid_divisions:")
print("   return divisions[-2] <= divisions[-1]")
print("")
print("   For empty list []:")
print("     - Accessing divisions[-2] when list is empty causes IndexError")
print("")
print("   For single-element list [1]:")
print("     - divisions has length 1")
print("     - Valid indices are 0 and -1 (both refer to the same element)")
print("     - divisions[-2] would be index -2, which doesn't exist")
print("     - This causes IndexError")

# Test 5: Test edge cases mentioned in docs
print("\n5. Testing documented examples:")
test_cases = [
    ([1, 2, 3], True, "ascending order"),
    ([3, 2, 1], False, "descending order"),
    ([1, 1, 1], False, "all equal"),
    ([0, 1, 1], True, "last two equal"),
    ((1, 2, 3), True, "tuple input"),
    (123, False, "non-list/tuple input"),
]

for divisions, expected, description in test_cases:
    try:
        result = valid_divisions(divisions)
        status = "PASS" if result == expected else f"FAIL (got {result})"
        print(f"   {divisions} -> {result} [{status}] ({description})")
    except Exception as e:
        print(f"   {divisions} -> ERROR: {e} ({description})")