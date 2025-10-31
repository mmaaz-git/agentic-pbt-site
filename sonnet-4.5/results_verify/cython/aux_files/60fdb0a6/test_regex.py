#!/usr/bin/env python3
"""Test to understand the regex splitting behavior"""

import re

# The regex used in _parse_pattern
pattern_regex = r"(?<!\\)/"

test_inputs = [
    "\\/",  # This is what happens when we strip the first "/" from "/\\/"
    "test/pattern",
    "\\//pattern",
    "a\\/b/pattern",
    "/pattern",
    "",
]

print("Testing regex: r\"(?<!\\\\)/\"")
print("This regex matches a '/' that is NOT preceded by '\\\\'")
print("=" * 60)

for test_input in test_inputs:
    print(f"\nInput: {repr(test_input)}")
    print(f"Raw bytes: {test_input.encode('unicode_escape').decode('ascii')}")

    # Test if the regex finds any matches
    matches = list(re.finditer(pattern_regex, test_input))
    if matches:
        print(f"Found {len(matches)} match(es):")
        for m in matches:
            print(f"  - Position {m.start()}: '{m.group()}'")
    else:
        print("No matches found")

    # Test the split
    split_result = re.split(pattern_regex, test_input, maxsplit=1)
    print(f"Split result: {split_result}")
    print(f"Number of parts: {len(split_result)}")

# Special analysis of the failing case
print("\n" + "=" * 60)
print("SPECIAL ANALYSIS: The failing case")
print("=" * 60)

failing_input = "/\\/"
print(f"Original input: {repr(failing_input)}")
print(f"After pattern[1:]: {repr(failing_input[1:])}")
substring = failing_input[1:]

print(f"\nAnalyzing substring: {repr(substring)}")
print(f"Character breakdown:")
for i, char in enumerate(substring):
    print(f"  Position {i}: {repr(char)} (ASCII {ord(char)})")

print(f"\nDoes the regex match? Let's check...")
matches = list(re.finditer(pattern_regex, substring))
if matches:
    print(f"Matches found: {matches}")
else:
    print("No matches found - this is why split returns only 1 element!")

print("\nThe regex r\"(?<!\\\\)/\" means:")
print("  - Match a forward slash '/'")
print("  - But only if it's NOT preceded by a backslash '\\\\'")
print(f"  - In the string {repr(substring)}, the '/' IS preceded by '\\\\'")
print("  - So the regex doesn't match, and split returns ['\\\\/'] (1 element)")
print("  - The code expects 2 elements and crashes with ValueError")