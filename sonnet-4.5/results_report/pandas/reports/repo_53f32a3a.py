#!/usr/bin/env python3
"""Demonstrate the _excel2num whitespace bug"""

from pandas.io.excel._util import _excel2num

# Test various whitespace inputs
test_cases = [
    ' ',      # single space
    '',       # empty string
    '\t',     # tab
    '\n',     # newline
    '   ',    # multiple spaces
    '\t\n',   # tab and newline
]

print("Testing whitespace inputs that should raise ValueError:")
print("=" * 60)

for test_input in test_cases:
    repr_input = repr(test_input)
    try:
        result = _excel2num(test_input)
        print(f"Input: {repr_input:<15} -> Result: {result} (BUG: should raise ValueError)")
    except ValueError as e:
        print(f"Input: {repr_input:<15} -> Raised ValueError: {e}")

print("\n" + "=" * 60)
print("For comparison, testing valid and invalid inputs:")
print("=" * 60)

# Test valid inputs
valid_cases = ['A', 'Z', 'AA', 'AB', 'XYZ']
for test_input in valid_cases:
    try:
        result = _excel2num(test_input)
        print(f"Input: {repr(test_input):<15} -> Result: {result} (valid)")
    except ValueError as e:
        print(f"Input: {repr(test_input):<15} -> Raised ValueError: {e}")

print()

# Test other invalid inputs that correctly raise errors
invalid_cases = ['A B', '123', 'A1', '!@#']
for test_input in invalid_cases:
    try:
        result = _excel2num(test_input)
        print(f"Input: {repr(test_input):<15} -> Result: {result}")
    except ValueError as e:
        print(f"Input: {repr(test_input):<15} -> Raised ValueError: {e} (correct)")