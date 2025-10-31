#!/usr/bin/env python3
"""Test script to reproduce the is_re_compilable bug with correct hypothesis format."""

import pandas.api.types as pt
import re
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

errors_found = []

@given(st.text())
@settings(max_examples=100, deadline=None)
def test_is_re_compilable_on_strings(s):
    # What the function returns
    try:
        result = pt.is_re_compilable(s)
    except Exception as e:
        errors_found.append((s, str(e)))
        raise AssertionError(f"is_re_compilable raised exception for {repr(s[:50])}: {e}")

    # What re.compile does
    try:
        re.compile(s)
        can_compile = True
    except (re.error, TypeError):
        can_compile = False

    # Check consistency
    assert result == can_compile, f"Mismatch for {repr(s[:50])}: is_re_compilable={result}, expected={can_compile}"

# Run the test
try:
    test_is_re_compilable_on_strings()
    print("Hypothesis test completed without finding issues")
except AssertionError as e:
    print(f"Hypothesis test found issue: {e}")

if errors_found:
    print(f"\nFound {len(errors_found)} inputs that caused exceptions:")
    for inp, err in errors_found[:5]:  # Show first 5
        print(f"  Input {repr(inp[:30])}: {err}")