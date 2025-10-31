#!/usr/bin/env python3
"""Verify Python's documented behavior for str.replace with empty strings"""

# Test what Python actually does with empty string replacements
print("=== Python's str.replace with empty string pattern ===")
print()
print("When replacing empty string '', Python inserts the replacement")
print("at every position between and around characters:")
print()

examples = [
    ('', 'X'),
    ('a', 'X'),
    ('ab', 'X'),
    ('abc', 'X'),
]

for s, repl in examples:
    result = s.replace('', repl)
    print(f"'{s}'.replace('', '{repl}') = '{result}'")
    if s:
        positions = [f"before '{s[0]}'"] + [f"between '{s[i]}' and '{s[i+1]}'" for i in range(len(s)-1)] + [f"after '{s[-1]}'"]
    else:
        positions = ["at the single position in empty string"]
    print(f"  Insertions at {len(s)+1} positions: {', '.join(positions)}")
    print()

# Verify this is standard Python behavior
print("=== Verification: This is standard, documented Python behavior ===")
print()
print("Python documentation states that str.replace replaces ALL occurrences.")
print("An empty string matches at every position, including:")
print("- Before the first character")
print("- Between each pair of characters")
print("- After the last character")
print()

# Test with count parameter
print("=== Testing with count parameter ===")
print()
for s in ['', 'abc']:
    for count in [0, 1, 2, -1]:
        result = s.replace('', 'X', count)
        print(f"'{s}'.replace('', 'X', count={count}) = '{result}'")