#!/usr/bin/env python3
"""Deeper analysis of the key_split behavior"""

import re

hex_pattern = re.compile("[a-f]+")

print("The key insight:")
print("=" * 50)
print()

print("The condition in line 1989-1990 is:")
print("if word.isalpha() and not (len(word) == 8 and hex_pattern.match(word) is not None):")
print("    result += '-' + word")
print("else:")
print("    break")
print()

print("This means a word is ADDED to the result if:")
print("1. word.isalpha() is True AND")
print("2. NOT (len(word) == 8 AND hex_pattern.match(word))")
print()
print("So a word causes a BREAK (stops being added) if:")
print("- word.isalpha() is False (contains non-alphabetic chars), OR")
print("- (len(word) == 8 AND hex_pattern.match(word)) when word is alphabetic")
print()

print("Let's verify with the examples:")
print()

examples = [
    ('task-abcdefab', 'abcdefab'),
    ('task-12345678', '12345678'),
    ('task-deadbeef', 'deadbeef'),
    ('task-1234abcd', '1234abcd'),
    ('task-00000000', '00000000'),
]

for full_key, suffix in examples:
    print(f"For suffix '{suffix}':")
    is_alpha = suffix.isalpha()
    has_hex_match = hex_pattern.match(suffix) is not None

    print(f"  - isalpha() = {is_alpha}")
    print(f"  - hex_pattern.match() = {has_hex_match}")

    if not is_alpha:
        print(f"  → Will BREAK because isalpha() is False (contains digits)")
    elif len(suffix) == 8 and has_hex_match:
        print(f"  → Will BREAK because it's 8 chars and matches [a-f]+")
    else:
        print(f"  → Will be ADDED to result")

    from dask.utils import key_split
    result = key_split(full_key)
    print(f"  Actual result: key_split('{full_key}') = '{result}'")
    print()

print("\nCONCLUSION:")
print("-" * 50)
print("The function DOES strip 8-character suffixes containing digits!")
print("It strips them because they fail the isalpha() test, NOT because")
print("of the hex pattern. The hex pattern only applies to pure alphabetic")
print("8-character suffixes like 'abcdefab' or 'deadbeef'.")
print()
print("The bug reporter is WRONG about the cause but RIGHT about the effect.")
print("The code works correctly but for a different reason than expected.")