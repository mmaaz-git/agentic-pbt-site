#!/usr/bin/env python3
"""Test script to reproduce the is_re_compilable bug."""

import pandas.api.types as pt
import re
import traceback

# Test the hypothesis test case
from hypothesis import given, strategies as st, settings

print("Testing is_re_compilable bug report...")
print("=" * 50)

# First, test the specific failing cases mentioned
test_cases = ['?', '*', '+', '(?', '[']

for test_input in test_cases:
    print(f"\nTesting with input: {repr(test_input)}")
    try:
        result = pt.is_re_compilable(test_input)
        print(f"  is_re_compilable returned: {result}")
    except Exception as e:
        print(f"  EXCEPTION raised: {type(e).__name__}: {e}")

    # Check what re.compile does
    print(f"  Testing re.compile directly:")
    try:
        re.compile(test_input)
        print(f"    re.compile succeeded")
    except re.error as e:
        print(f"    re.compile raised re.error: {e}")
    except Exception as e:
        print(f"    re.compile raised {type(e).__name__}: {e}")

print("\n" + "=" * 50)
print("Running hypothesis test...")

@given(st.text())
@settings(max_examples=300)
def test_is_re_compilable_on_strings(s):
    # What the function returns
    try:
        result = pt.is_re_compilable(s)
        func_raised = False
    except:
        func_raised = True
        result = None

    # What re.compile does
    try:
        re.compile(s)
        can_compile = True
    except re.error:
        can_compile = False
    except TypeError:
        can_compile = False

    # Check consistency
    if func_raised:
        print(f"FUNCTION RAISED EXCEPTION for input: {repr(s[:50])}")
        return False

    if result != can_compile:
        print(f"MISMATCH for input: {repr(s[:50])}")
        print(f"  is_re_compilable returned: {result}")
        print(f"  re.compile would succeed: {can_compile}")
        return False

    return True

# Run the test
try:
    test_is_re_compilable_on_strings()
    print("Hypothesis test completed without finding issues")
except Exception as e:
    print(f"Hypothesis test failed: {e}")
    traceback.print_exc()