#!/usr/bin/env python3
"""Test Python's native string concatenation behavior with null characters."""

print("Testing Python's native string concatenation with null characters:")
print("="*60)

test_cases = [
    ('\x00', 'test'),
    ('\x00\x00', 'abc'),
    ('a\x00', 'b'),
    ('a\x00b', 'c'),
    ('\x00a', 'b'),
    ('test', '\x00'),
    ('\x00', '\x00'),
]

for s1, s2 in test_cases:
    result = s1 + s2
    print(f"'{repr(s1)}' + '{repr(s2)}' = '{repr(result)}'")
    # Verify length preservation
    expected_len = len(s1) + len(s2)
    actual_len = len(result)
    print(f"  Length check: {actual_len} == {expected_len}: {actual_len == expected_len}")
    # Verify character preservation
    print(f"  Contains all chars: {all(c in result for c in s1+s2)}")
    print()

print("\nConclusion: Python preserves all characters including null bytes in string concatenation.")