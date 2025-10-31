#!/usr/bin/env python3
"""Test the reported bug in is_body_allowed_for_status_code"""

import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
from fastapi.utils import is_body_allowed_for_status_code

# First, let's test the basic functionality with valid inputs
print("=== Testing Valid Inputs ===")
valid_tests = [
    (None, True, "None should return True"),
    ("default", True, "default should return True"),
    ("1XX", True, "1XX should return True"),
    ("2XX", True, "2XX should return True"),
    ("3XX", True, "3XX should return True"),
    ("4XX", True, "4XX should return True"),
    ("5XX", True, "5XX should return True"),
    (200, True, "200 should return True (body allowed)"),
    (201, True, "201 should return True (body allowed)"),
    (204, False, "204 should return False (no body)"),
    (205, False, "205 should return False (no body)"),
    (304, False, "304 should return False (no body)"),
    (199, False, "199 should return False (1xx no body)"),
    (100, False, "100 should return False (1xx no body)"),
    ("204", False, "String '204' should return False"),
    ("200", True, "String '200' should return True"),
]

for test_input, expected, description in valid_tests:
    try:
        result = is_body_allowed_for_status_code(test_input)
        assert result == expected, f"Failed: {description}. Got {result}, expected {expected}"
        print(f"✓ {description}")
    except Exception as e:
        print(f"✗ {description}: {e}")

print("\n=== Testing Bug Report Case ===")
# Test the specific bug case from the report
try:
    result = is_body_allowed_for_status_code("invalid")
    print(f"✗ Expected ValueError but got result: {result}")
except ValueError as e:
    print(f"✓ Confirmed bug: ValueError raised as reported: {e}")
except Exception as e:
    print(f"? Unexpected error: {e}")

print("\n=== Testing Other Invalid String Cases ===")
invalid_strings = ["abc", "error", "", "foo", "NaN", "Infinity", "-1", "6XX", "10X", "ABCD"]

for invalid_str in invalid_strings:
    try:
        result = is_body_allowed_for_status_code(invalid_str)
        print(f"✗ '{invalid_str}' - Expected ValueError but got result: {result}")
    except ValueError as e:
        print(f"✓ '{invalid_str}' - ValueError raised: {e}")
    except Exception as e:
        print(f"? '{invalid_str}' - Unexpected error: {e}")

print("\n=== Running Property-Based Test ===")
# Run the hypothesis test from the bug report
valid_patterns = {"default", "1XX", "2XX", "3XX", "4XX", "5XX"}

@given(st.text())
def test_is_body_allowed_for_status_code_handles_all_strings(status_code):
    assume(status_code not in valid_patterns)

    try:
        result = is_body_allowed_for_status_code(status_code)
        assert isinstance(result, bool), f"Result should be bool, got {type(result)}"
        # If we get here, the function handled the string without error
        return True
    except ValueError:
        # This is the bug - ValueError is raised for invalid strings
        return False

# Run property test on a sample of strings
print("Testing random strings with Hypothesis...")
test_strings = ["x", "test", "???", "12.34", "1e10", "true", "false", "null"]
failures = []
for test_str in test_strings:
    try:
        result = is_body_allowed_for_status_code(test_str)
        print(f"  '{test_str}' -> {result} (no error)")
    except ValueError as e:
        failures.append(test_str)
        print(f"  '{test_str}' -> ValueError")

if failures:
    print(f"\n✓ Bug confirmed: {len(failures)} strings caused ValueError")
else:
    print("\n✗ Bug not reproduced with test strings")

print("\n=== Edge Cases ===")
edge_cases = [
    ("", "empty string"),
    ("  ", "whitespace"),
    ("\n", "newline"),
    ("None", "string 'None'"),
    ("null", "string 'null'"),
    ("true", "string 'true'"),
    ("false", "string 'false'"),
    ("200.0", "float string"),
    ("1e2", "scientific notation"),
    ("0x64", "hex notation"),
    ("NaN", "NaN string"),
    ("Infinity", "Infinity string"),
    ("-200", "negative number string"),
]

for test_input, description in edge_cases:
    try:
        result = is_body_allowed_for_status_code(test_input)
        print(f"  {description}: '{test_input}' -> {result}")
    except ValueError as e:
        print(f"  {description}: '{test_input}' -> ValueError: {str(e)[:50]}...")
    except Exception as e:
        print(f"  {description}: '{test_input}' -> {type(e).__name__}: {str(e)[:50]}...")