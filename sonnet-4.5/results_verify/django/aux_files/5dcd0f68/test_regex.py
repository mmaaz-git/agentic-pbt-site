#!/usr/bin/env python3
import re

# This is the pattern used in the code
pattern = re.compile(r"[A-Z_]+")

# Valid MySQL EXTRACT unit specifiers from documentation
valid_units = [
    "YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND",
    "MICROSECOND", "YEAR_MONTH", "DAY_MINUTE", "DAY_HOUR",
    "DAY_SECOND", "DAY_MICROSECOND", "HOUR_MINUTE",
    "HOUR_SECOND", "HOUR_MICROSECOND", "MINUTE_SECOND",
    "MINUTE_MICROSECOND", "SECOND_MICROSECOND", "QUARTER", "WEEK"
]

print("Testing valid MySQL EXTRACT units:")
for unit in valid_units:
    if pattern.fullmatch(unit):
        print(f"  ✓ {unit} - matches regex")
    else:
        print(f"  ✗ {unit} - does NOT match regex")

print("\nTesting some invalid inputs:")
invalid = ["year", "YEAR123", "123YEAR", "YE AR", "YE@R", ""]
for unit in invalid:
    if pattern.fullmatch(unit):
        print(f"  ✗ {unit} - matches regex (shouldn't)")
    else:
        print(f"  ✓ {unit} - does NOT match regex (correct)")