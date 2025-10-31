#!/usr/bin/env python3
"""Verify that the existing test cases in TestCythonUtils would catch this bug"""

from Cython.Utils import normalise_float_repr

# Test cases from the existing test suite that are similar
test_cases_from_suite = [
    ('1.1E-5', '.000011'),
    ('12.3E-5', '.000123'),
]

print("Testing existing test cases from TestCythonUtils.py:")
print("="*60)

for input_str, expected in test_cases_from_suite:
    result = normalise_float_repr(input_str)
    float_input = float(input_str)
    try:
        float_result = float(result)
        preserved = abs(float_input - float_result) < 1e-10
        matches_expected = (result == expected)

        print(f"Input: {input_str}")
        print(f"  Expected string: {expected}")
        print(f"  Got string:      {result}")
        print(f"  Matches expected: {matches_expected}")
        print(f"  Float preserved:  {preserved}")
        if not preserved:
            print(f"    Input float:  {float_input}")
            print(f"    Result float: {float_result}")
            print(f"    Difference:   {abs(float_input - float_result)}")
        print()
    except Exception as e:
        print(f"  ERROR: {e}")
        print()