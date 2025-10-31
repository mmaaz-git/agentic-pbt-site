#!/usr/bin/env python3
"""Minimal reproduction of attrs.converters.to_bool float acceptance bug"""

from attrs import converters

# Test cases that should raise ValueError according to documentation
# but actually succeed (BUG)
print("Testing float values that should raise ValueError:")
print("-" * 50)

try:
    result = converters.to_bool(0.0)
    print(f"converters.to_bool(0.0) = {result} (BUG: Should raise ValueError)")
except ValueError as e:
    print(f"converters.to_bool(0.0) raised ValueError: {e}")

try:
    result = converters.to_bool(1.0)
    print(f"converters.to_bool(1.0) = {result} (BUG: Should raise ValueError)")
except ValueError as e:
    print(f"converters.to_bool(1.0) raised ValueError: {e}")

print("\n" + "=" * 50 + "\n")

# Test other float values to show inconsistency
print("Testing other float values (correctly raise ValueError):")
print("-" * 50)

test_values = [0.5, 1.5, 2.0, -1.0, 10.0]
for val in test_values:
    try:
        result = converters.to_bool(val)
        print(f"converters.to_bool({val}) = {result} (BUG: Should raise ValueError)")
    except ValueError as e:
        print(f"converters.to_bool({val}) correctly raised ValueError")

print("\n" + "=" * 50 + "\n")

# Demonstrate why this happens
print("Root cause - Python's equality behavior:")
print("-" * 50)
print(f"0.0 == 0: {0.0 == 0}")
print(f"1.0 == 1: {1.0 == 1}")
print(f"0.0 in (0,): {0.0 in (0,)}")
print(f"1.0 in (1,): {1.0 in (1,)}")

print("\n" + "=" * 50 + "\n")

# Show documented valid inputs for comparison
print("Documented valid inputs (from docstring):")
print("-" * 50)
print("Values mapping to True: True, 'true'/'t', 'yes'/'y', 'on', '1', 1")
print("Values mapping to False: False, 'false'/'f', 'no'/'n', 'off', '0', 0")
print("\nNote: The documentation lists integer 1 and 0, NOT float 1.0 and 0.0")