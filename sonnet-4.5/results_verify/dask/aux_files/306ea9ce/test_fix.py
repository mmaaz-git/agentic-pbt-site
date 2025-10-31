#!/usr/bin/env python3
"""Test the proposed fix for the unquote function"""

from dask.core import istask

def unquote_original(expr):
    """Original implementation with the bug"""
    if istask(expr):
        if expr[0] in (tuple, list, set):
            return expr[0](map(unquote_original, expr[1]))
        elif (
            expr[0] == dict
            and isinstance(expr[1], list)
            and isinstance(expr[1][0], list)  # BUG: IndexError when expr[1] is empty
        ):
            return dict(map(unquote_original, expr[1]))
    return expr

def unquote_fixed(expr):
    """Fixed implementation"""
    if istask(expr):
        if expr[0] in (tuple, list, set):
            return expr[0](map(unquote_fixed, expr[1]))
        elif (
            expr[0] == dict
            and isinstance(expr[1], list)
            and (not expr[1] or isinstance(expr[1][0], list))  # FIX: Check if empty first
        ):
            return dict(map(unquote_fixed, expr[1]))
    return expr

# Test cases
test_cases = [
    ("Empty dict task", (dict, []), {}),
    ("Dict with list pairs", (dict, [['a', 1], ['b', 2]]), {'a': 1, 'b': 2}),
    ("Dict with nested lists", (dict, [['a', [1, 2, 3]], ['b', 2]]), {'a': [1, 2, 3], 'b': 2}),
    ("Empty tuple task", (tuple, []), ()),
    ("Empty list task", (list, []), []),
    ("Empty set task", (set, []), set()),
    ("Regular list (not a task)", [1, 2, 3], [1, 2, 3]),
    ("Regular dict (not a task)", {'a': 1}, {'a': 1}),
]

print("Testing original vs fixed implementation:\n")
for description, input_val, expected in test_cases:
    print(f"Test: {description}")
    print(f"Input: {input_val}")
    print(f"Expected: {expected}")

    # Test original
    try:
        result_orig = unquote_original(input_val)
        print(f"Original: {result_orig} ✓")
    except Exception as e:
        print(f"Original: ERROR - {type(e).__name__}: {e}")

    # Test fixed
    try:
        result_fixed = unquote_fixed(input_val)
        if result_fixed == expected:
            print(f"Fixed: {result_fixed} ✓")
        else:
            print(f"Fixed: {result_fixed} (expected {expected}) ✗")
    except Exception as e:
        print(f"Fixed: ERROR - {type(e).__name__}: {e}")

    print()