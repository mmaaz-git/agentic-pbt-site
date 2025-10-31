#!/usr/bin/env python3
"""Test convert_json_to_lines behavior"""

from pandas._libs.writers import convert_json_to_lines

test_inputs = [
    "",
    "1, 2, 3",
    '"a","b"',
    '{"foo": "bar"}',
    "1",
    "null",
]

print("Testing convert_json_to_lines:")
print("=" * 50)

for inp in test_inputs:
    try:
        result = convert_json_to_lines(inp)
        print(f"Input: {inp!r:<20} -> Output: {result!r}")
    except Exception as e:
        print(f"Input: {inp!r:<20} -> Error: {e}")