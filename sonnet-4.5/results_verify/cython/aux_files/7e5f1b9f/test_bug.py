#!/usr/bin/env python3

# Test 1: Run the hypothesis test with the failing input
from hypothesis import given, strategies as st
from Cython.Plex.Regexps import chars_to_ranges

def test_chars_to_ranges_preserves_all_characters(s):
    ranges = chars_to_ranges(s)

    assert len(ranges) % 2 == 0

    reconstructed_chars = set()
    for i in range(0, len(ranges), 2):
        code1, code2 = ranges[i], ranges[i + 1]
        for code in range(code1, code2):
            reconstructed_chars.add(chr(code))

    assert reconstructed_chars == set(s), f"Expected {set(s)}, got {reconstructed_chars}"

# Test with the reported failing input
print("Testing with s='00'...")
try:
    test_chars_to_ranges_preserves_all_characters('00')
    print("Test passed with '00' (unexpected)")
except AssertionError as e:
    print(f"Test failed with '00' (as expected): {e}")

# Test 2: Low-level reproduction
print("\n--- Low-level reproduction ---")
s = '00'
ranges = chars_to_ranges(s)
print(f"Input string: {repr(s)}")
print(f"Set of input characters: {set(s)}")
print(f"Returned ranges: {ranges}")

reconstructed = set()
for i in range(0, len(ranges), 2):
    code1, code2 = ranges[i], ranges[i + 1]
    print(f"Range [{code1}, {code2}): characters {repr(chr(code1))} to {repr(chr(code2-1))}")
    for code in range(code1, code2):
        reconstructed.add(chr(code))

print(f"Reconstructed set: {reconstructed}")
print(f"Expected set: {{'0'}}")
print(f"Bug confirmed: {'0' in reconstructed and '1' in reconstructed}")

# Test 3: High-level impact with Any()
print("\n--- High-level impact with Any() ---")
from io import StringIO
from Cython.Plex import Any, Lexicon, Scanner, TEXT

try:
    lexicon = Lexicon([(Any('00'), TEXT)])

    # Test if '1' is matched (it shouldn't be)
    scanner = Scanner(lexicon, StringIO('1'))
    value, text = scanner.read()
    print(f"Any('00') matched '1': {text == '1'}")
    if text == '1':
        print("Bug confirmed: Any('00') incorrectly matches '1'")

    # Test if '0' is matched (it should be)
    scanner2 = Scanner(lexicon, StringIO('0'))
    value2, text2 = scanner2.read()
    print(f"Any('00') matched '0': {text2 == '0'}")

except Exception as e:
    print(f"Error in high-level test: {e}")