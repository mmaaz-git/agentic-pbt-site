#!/usr/bin/env python3
"""Test if the proposed fix actually works when implemented correctly"""

def chars_to_ranges_current(s):
    """Current implementation from Cython"""
    char_list = list(s)
    char_list.sort()
    i = 0
    n = len(char_list)
    result = []
    while i < n:
        code1 = ord(char_list[i])
        code2 = code1 + 1
        i += 1
        while i < n and code2 >= ord(char_list[i]):
            code2 += 1
            i += 1
        result.append(code1)
        result.append(code2)
    return result

def chars_to_ranges_proper_fix(s):
    """A proper fix that handles duplicates correctly"""
    # First remove duplicates by converting to set
    unique_chars = set(s)
    char_list = sorted(unique_chars)

    i = 0
    n = len(char_list)
    result = []
    while i < n:
        code1 = ord(char_list[i])
        code2 = code1 + 1
        i += 1
        # Now we can safely merge consecutive characters
        while i < n and code2 >= ord(char_list[i]):
            code2 += 1
            i += 1
        result.append(code1)
        result.append(code2)
    return result

def chars_to_ranges_alternative_fix(s):
    """Alternative fix: skip duplicates in the inner loop"""
    char_list = list(s)
    char_list.sort()
    i = 0
    n = len(char_list)
    result = []
    while i < n:
        code1 = ord(char_list[i])
        code2 = code1 + 1
        i += 1
        # Skip duplicates
        while i < n and ord(char_list[i]) == code1:
            i += 1
        # Then merge consecutive characters
        while i < n and code2 >= ord(char_list[i]):
            code2 += 1
            i += 1
        result.append(code1)
        result.append(code2)
    return result

# Test all three versions
test_cases = ['00', 'aa', '11', 'aaa', '01', 'ab', 'abc', 'zz', '99']

for version_name, func in [
    ("CURRENT (BUGGY)", chars_to_ranges_current),
    ("PROPER FIX (deduplicate)", chars_to_ranges_proper_fix),
    ("ALTERNATIVE FIX (skip dups)", chars_to_ranges_alternative_fix)
]:
    print(f"\n{'='*60}")
    print(f"{version_name}")
    print('='*60)

    all_correct = True
    for s in test_cases:
        result = func(s)
        covered = set()
        for i in range(0, len(result), 2):
            for code in range(result[i], result[i + 1]):
                covered.add(chr(code))

        expected = set(s)
        correct = covered == expected
        if not correct:
            all_correct = False

        status = "✓" if correct else "✗"
        print(f"{status} '{s}': Expected {expected}, Got {covered}")

    print(f"\nAll tests passed: {all_correct}")