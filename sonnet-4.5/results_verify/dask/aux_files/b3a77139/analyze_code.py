#!/usr/bin/env python3
"""Analyze the key_split code implementation"""

import re

# This is the pattern from the code
hex_pattern = re.compile("[a-f]+")

# Let's test what this pattern matches
test_strings = [
    'abcdefab',  # All letters
    '12345678',  # All digits
    '1234abcd',  # Mixed digits and letters
    'abcd1234',  # Mixed letters and digits
    '00000000',  # All zeros
    'ffffffff',  # All f's
    'deadbeef',  # Classic hex pattern
]

print("Testing hex_pattern = re.compile('[a-f]+'):")
for s in test_strings:
    match = hex_pattern.match(s)
    if match:
        print(f"  '{s}': MATCHED (match object: {match.group()})")
    else:
        print(f"  '{s}': NO MATCH")

print("\nChecking the condition from line 1989-1990:")
print("word.isalpha() and not (len(word) == 8 and hex_pattern.match(word) is not None)")
print()

for word in test_strings:
    is_alpha = word.isalpha()
    is_8_chars = len(word) == 8
    has_hex_match = hex_pattern.match(word) is not None
    condition = is_alpha and not (is_8_chars and has_hex_match)

    print(f"word='{word}':")
    print(f"  isalpha={is_alpha}, len==8={is_8_chars}, hex_match={has_hex_match}")
    print(f"  condition evaluates to: {condition}")
    print()

# Let's trace through the logic for specific examples
print("\nTracing key_split logic for specific examples:")
print("=" * 50)

def trace_key_split(s):
    print(f"\nTracing: key_split('{s}')")

    words = s.split("-")
    print(f"  words = {words}")

    if not words[0][0].isalpha():
        result = words[0].split(",")[0].strip("_'()\"")
        print(f"  First char not alpha, result = '{result}'")
    else:
        result = words[0]
        print(f"  First word: result = '{result}'")

    for i, word in enumerate(words[1:], 1):
        print(f"  Processing word[{i}] = '{word}':")
        is_alpha = word.isalpha()
        is_8_chars = len(word) == 8
        has_hex_match = hex_pattern.match(word) is not None

        print(f"    isalpha={is_alpha}, len==8={is_8_chars}, hex_match={has_hex_match}")

        if is_alpha and not (is_8_chars and has_hex_match):
            result += "-" + word
            print(f"    Adding word, result = '{result}'")
        else:
            print(f"    Breaking, final result = '{result}'")
            break

    return result

# Test the examples from bug report
trace_key_split('task-abcdefab')
trace_key_split('task-12345678')
trace_key_split('task-deadbeef')