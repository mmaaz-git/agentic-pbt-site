#!/usr/bin/env python3
"""Test the capitalize_first_letter bug report"""

# First test: The hypothesis test
from hypothesis import given, strategies as st
from pandas.util import capitalize_first_letter

@given(st.text())
def test_capitalize_first_letter_length_preserved(s):
    result = capitalize_first_letter(s)
    assert len(result) == len(s), f"Input: {s!r}, Output: {result!r}"

# Run the hypothesis test with the failing example
print("Testing with Hypothesis...")
try:
    test_capitalize_first_letter_length_preserved()
    print("Hypothesis test passed (unexpected)")
except AssertionError as e:
    print(f"Hypothesis test failed as expected: {e}")

# Second test: Direct reproduction with the specific case
print("\n" + "="*50)
print("Direct reproduction with 'ß':")
print("="*50)

s = 'ß'
result = capitalize_first_letter(s)
print(f"Input: {s!r} (length {len(s)})")
print(f"Output: {result!r} (length {len(result)})")

try:
    assert len(result) == len(s), f"Expected length {len(s)}, got {len(result)}"
    print("Assertion passed (unexpected)")
except AssertionError as e:
    print(f"Assertion failed as expected: {e}")

# Test other Unicode characters
print("\n" + "="*50)
print("Testing other Unicode characters:")
print("="*50)

test_cases = [
    'ß',  # German eszett
    'ﬃ',  # ffi ligature
    'ﬄ',  # ffl ligature
    'ﬅ',  # st ligature
    'ı',  # Turkish lowercase dotless i
    'a',  # Regular ASCII
    '',   # Empty string
    '123', # Numbers
    'Hello', # Already capitalized
    'ßtest', # eszett at start
]

for test_str in test_cases:
    result = capitalize_first_letter(test_str)
    length_preserved = len(result) == len(test_str)
    print(f"Input: {test_str!r:10} -> Output: {result!r:10} | Length: {len(test_str)} -> {len(result)} | Preserved: {length_preserved}")

# Test how Python's upper() behaves with these characters
print("\n" + "="*50)
print("Understanding Python's .upper() behavior:")
print("="*50)

for char in ['ß', 'ﬃ', 'ﬄ', 'ﬅ', 'ı']:
    upper_char = char.upper()
    print(f"'{char}'.upper() = '{upper_char}' (length {len(char)} -> {len(upper_char)})")