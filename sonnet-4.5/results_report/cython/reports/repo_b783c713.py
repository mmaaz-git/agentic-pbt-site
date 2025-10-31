#!/usr/bin/env python3
"""Demonstrating the chars_to_ranges bug with duplicate characters."""

from Cython.Plex.Regexps import chars_to_ranges
from io import StringIO
from Cython.Plex import Any, Lexicon, Scanner, TEXT

# Low-level demonstration of the bug
print("=== Low-level demonstration ===")
print("Testing chars_to_ranges('00')...")

s = '00'
ranges = chars_to_ranges(s)
print(f"Input string: {repr(s)}")
print(f"Returned ranges: {ranges}")

# Reconstruct characters from ranges
reconstructed = set()
for i in range(0, len(ranges), 2):
    code1, code2 = ranges[i], ranges[i + 1]
    for code in range(code1, code2):
        reconstructed.add(chr(code))

print(f"Reconstructed characters: {reconstructed}")
print(f"Expected characters: {set(s)}")

# Check if they match
if reconstructed != set(s):
    print(f"ERROR: Expected {set(s)}, but got {reconstructed}")
    print(f"The function incorrectly includes character(s): {reconstructed - set(s)}")
else:
    print("OK: Reconstructed characters match input")

print()

# High-level demonstration showing impact on public API
print("=== High-level impact on Any() function ===")
print("Creating lexicon with Any('00')...")

lexicon = Lexicon([(Any('00'), TEXT)])
print("Any('00') should match only '0', but let's test if it matches '1'...")

scanner = Scanner(lexicon, StringIO('1'))
value, text = scanner.read()

if text == '1':
    print(f"ERROR: Any('00') incorrectly matched '1'")
    print(f"This demonstrates that the bug affects the public API")
else:
    print(f"Any('00') did not match '1' (unexpected)")

print()
print("=== Explanation ===")
print("The bug occurs because chars_to_ranges uses '>=' instead of '>' when")
print("checking if consecutive characters should be merged into a range.")
print("With duplicate '0' characters, it incorrectly expands the range to include '1'.")