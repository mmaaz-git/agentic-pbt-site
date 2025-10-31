#!/usr/bin/env python3
"""Test script to reproduce the reported bug"""

from fastapi.utils import is_body_allowed_for_status_code

# Test 1: Basic reproduction with non-numeric strings
print("Test 1: Non-numeric string 'abc'")
try:
    result = is_body_allowed_for_status_code("abc")
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

print("\nTest 2: Decimal string '200.0'")
try:
    result = is_body_allowed_for_status_code("200.0")
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

print("\nTest 3: Empty string ''")
try:
    result = is_body_allowed_for_status_code("")
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

print("\nTest 4: String with spaces '200 OK'")
try:
    result = is_body_allowed_for_status_code("200 OK")
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

# Test valid cases for comparison
print("\nTest 5: Valid cases for comparison")
test_cases = [None, 200, "200", "2XX", "default", 204, 304]
for test in test_cases:
    try:
        result = is_body_allowed_for_status_code(test)
        print(f"  {test!r}: {result}")
    except Exception as e:
        print(f"  {test!r}: Error - {e}")