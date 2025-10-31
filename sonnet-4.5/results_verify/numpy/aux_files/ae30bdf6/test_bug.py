#!/usr/bin/env python3
"""Test the reported numpy.char.lower Unicode bug"""

import numpy as np
import numpy.char as char

# Test the specific case mentioned in the bug report
print("Testing Turkish capital İ (U+0130)")
print("=" * 50)

# The Turkish capital 'İ' character
test_char = 'İ'
print(f"Input character: {test_char!r} (Unicode: U+{ord(test_char):04X})")
print(f"Python str.lower() result: {test_char.lower()!r}")
print(f"Length of Python lowercase: {len(test_char.lower())}")

# Test with NumPy
arr = np.array([test_char], dtype=str)
print(f"\nNumPy array dtype: {arr.dtype}")
print(f"NumPy array content: {arr}")

# Apply numpy.char.lower
result = char.lower(arr)
print(f"\nAfter numpy.char.lower:")
print(f"Result dtype: {result.dtype}")
print(f"Result content: {result}")
print(f"Result[0]: {result[0]!r}")
print(f"Length of result[0]: {len(result[0])}")

# Check if they match
python_lower = test_char.lower()
numpy_lower = result[0]
print(f"\nComparison:")
print(f"Python result: {python_lower!r}")
print(f"NumPy result: {numpy_lower!r}")
print(f"Are they equal? {python_lower == numpy_lower}")

# Let's check the actual Unicode codepoints
print(f"\nCodepoint analysis:")
print(f"Python result codepoints: {[f'U+{ord(c):04X}' for c in python_lower]}")
print(f"NumPy result codepoints: {[f'U+{ord(c):04X}' for c in numpy_lower]}")

# Test with hypothesis example
print("\n" + "=" * 50)
print("Running hypothesis test case:")
from hypothesis import given, strategies as st

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=0x100, max_codepoint=0x1000, blacklist_categories=('Cs',)), min_size=1, max_size=10), min_size=1, max_size=10))
def test_upper_lower_unicode(strings):
    arr = np.array(strings, dtype=str)
    lower_result = char.lower(arr)

    for i in range(len(strings)):
        if lower_result[i] != strings[i].lower():
            print(f"Found mismatch at index {i}:")
            print(f"  Input: {strings[i]!r}")
            print(f"  Expected: {strings[i].lower()!r}")
            print(f"  Got: {lower_result[i]!r}")
            assert False, f"Mismatch found"

# Run with the specific failing input
# The hypothesis test takes no arguments when called directly
strings_to_test = ['İ']
arr = np.array(strings_to_test, dtype=str)
lower_result = char.lower(arr)
if lower_result[0] != strings_to_test[0].lower():
    print(f"Test failed as expected:")
    print(f"  Input: {strings_to_test[0]!r}")
    print(f"  Expected: {strings_to_test[0].lower()!r}")
    print(f"  Got: {lower_result[0]!r}")
else:
    print("Test passed (unexpected)")

# Test a few more expanding Unicode cases
print("\n" + "=" * 50)
print("Testing other potentially expanding Unicode cases:")

test_cases = [
    'İ',  # Turkish capital I with dot
    'ß',  # German sharp s (expands to 'ss' in uppercase but not lowercase)
    'Ω',  # Greek capital omega
    'Σ',  # Greek capital sigma
]

for test_str in test_cases:
    arr = np.array([test_str], dtype=str)
    result = char.lower(arr)
    expected = test_str.lower()
    match = result[0] == expected
    print(f"{test_str!r} -> NumPy: {result[0]!r}, Python: {expected!r}, Match: {match}")