#!/usr/bin/env python3
"""Test to reproduce the bug in pandas.io.json._normalize.convert_to_line_delimits"""

import pandas.io.json._normalize as normalize

print("Testing convert_to_line_delimits function")
print("=" * 50)

# Test cases from the bug report
test_cases = [
    ("00", "00"),
    ('{"foo": "bar"}', '{"foo": "bar"}'),
    ("x]", "x]"),
    ("[1, 2, 3]", "[1, 2, 3]"),
]

print("\nRunning test cases:")
for input_str, expected in test_cases:
    result = normalize.convert_to_line_delimits(input_str)
    if result != expected:
        print(f"BUG: {input_str!r} -> {result!r} (expected {expected!r})")
    else:
        print(f"OK: {input_str!r} -> {result!r}")

# Additional test to understand the function behavior
print("\n" + "=" * 50)
print("Additional test cases to understand behavior:")

additional_tests = [
    ("[", "["),  # Only starts with [
    ("]", "]"),  # Only ends with ]
    ("a]", "a]"),  # Ends with ] but doesn't start with [
    ("[a", "[a"),  # Starts with [ but doesn't end with ]
    ("[]", "[]"),  # Empty JSON array
    ('["a","b"]', '["a","b"]'),  # Valid JSON array with strings
]

for input_str, desc in additional_tests:
    try:
        result = normalize.convert_to_line_delimits(input_str)
        print(f"Input: {input_str!r} -> Output: {result!r}")
    except Exception as e:
        print(f"Input: {input_str!r} -> Error: {e}")