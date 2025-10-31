#!/usr/bin/env python3
"""Test the exact fix proposed in the bug report"""

def chars_to_ranges_original(s):
    """Original code with >= on line 43"""
    char_list = list(s)
    char_list.sort()
    i = 0
    n = len(char_list)
    result = []
    while i < n:
        code1 = ord(char_list[i])
        code2 = code1 + 1
        i += 1
        while i < n and code2 >= ord(char_list[i]):  # Line 43: Original
            code2 += 1
            i += 1
        result.append(code1)
        result.append(code2)
    return result

def chars_to_ranges_bug_report_fix(s):
    """Bug report's proposed fix: change >= to > on line 43"""
    char_list = list(s)
    char_list.sort()
    i = 0
    n = len(char_list)
    result = []
    while i < n:
        code1 = ord(char_list[i])
        code2 = code1 + 1
        i += 1
        while i < n and code2 > ord(char_list[i]):  # Line 43: Changed to >
            code2 += 1
            i += 1
        result.append(code1)
        result.append(code2)
    return result

# Test cases
test_cases = [
    ('00', {'0'}),           # Duplicate characters
    ('aa', {'a'}),           # Duplicate letters
    ('aaa', {'a'}),          # Triple duplicate
    ('01', {'0', '1'}),      # Consecutive digits
    ('ab', {'a', 'b'}),      # Consecutive letters
    ('abc', {'a', 'b', 'c'}),# Multiple consecutive
    ('ac', {'a', 'c'}),      # Non-consecutive
    ('acb', {'a', 'b', 'c'}),# Out of order
    ('0011', {'0', '1'}),    # Mixed duplicates and consecutive
]

print("Testing Original vs Bug Report's Proposed Fix")
print("=" * 70)

original_failures = 0
fix_failures = 0

for input_str, expected in test_cases:
    # Test original
    result_orig = chars_to_ranges_original(input_str)
    covered_orig = set()
    for i in range(0, len(result_orig), 2):
        for code in range(result_orig[i], result_orig[i + 1]):
            covered_orig.add(chr(code))
    orig_correct = covered_orig == expected

    # Test fix
    result_fix = chars_to_ranges_bug_report_fix(input_str)
    covered_fix = set()
    for i in range(0, len(result_fix), 2):
        for code in range(result_fix[i], result_fix[i + 1]):
            covered_fix.add(chr(code))
    fix_correct = covered_fix == expected

    if not orig_correct:
        original_failures += 1
    if not fix_correct:
        fix_failures += 1

    print(f"\nInput: '{input_str}'")
    print(f"  Expected: {expected}")
    print(f"  Original (>=): {covered_orig} {'✓' if orig_correct else '✗'}")
    print(f"  Fix (>):      {covered_fix} {'✓' if fix_correct else '✗'}")

    if not orig_correct and fix_correct:
        print(f"  >>> FIX WORKS FOR THIS CASE")

print("\n" + "=" * 70)
print(f"SUMMARY:")
print(f"  Original failures: {original_failures}/{len(test_cases)}")
print(f"  Fix failures:      {fix_failures}/{len(test_cases)}")

if fix_failures == 0:
    print("\n✓ THE BUG REPORT'S FIX APPEARS TO WORK!")
else:
    print(f"\n✗ THE BUG REPORT'S FIX STILL HAS {fix_failures} FAILURES")