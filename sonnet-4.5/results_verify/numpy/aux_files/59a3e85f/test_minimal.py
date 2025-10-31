#!/usr/bin/env python3
"""Minimal reproduction of the bug"""

import numpy.f2py.crackfortran as crackfortran

# Test various unbalanced parentheses cases
test_cases = [
    ')',
    '((',
    '(()',
    '())',
    ')))',
    '(a,b))',
    'func(arg))',
]

for test_input in test_cases:
    print(f"\nTesting: {test_input!r}")
    try:
        result = crackfortran.markoutercomma(test_input)
        print(f"  Result: {result!r}")
    except AssertionError as e:
        print(f"  AssertionError: {e}")
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")