#!/usr/bin/env python3
"""Test script to reproduce the pandas.eval empty expression bug."""

import sys
import traceback
from hypothesis import given, strategies as st
import pytest

# Import pandas eval
from pandas.core.computation.eval import eval

print("=" * 60)
print("Testing pandas.eval empty expression handling")
print("=" * 60)

# Test 1: Empty string
print("\nTest 1: eval('') - empty string")
try:
    result = eval("")
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

# Test 2: Whitespace strings
whitespace_cases = [
    ("'   '", "   "),  # Three spaces
    ("' '", " "),      # Single space
    ("'\\t'", "\t"),   # Tab
    ("'\\n'", "\n"),   # Newline
    ("'\\r\\n'", "\r\n"),  # Windows newline
    ("'  \\n  '", "  \n  "),  # Mixed whitespace
]

for display, test_str in whitespace_cases:
    print(f"\nTest: eval({display})")
    try:
        result = eval(test_str)
        print(f"Result: {repr(result)}")
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Running Hypothesis test")
print("=" * 60)

# Property-based test as provided in bug report
@given(st.text())
def test_empty_expressions_should_raise(s):
    """Test that all whitespace-only expressions raise ValueError."""
    if not s.strip():  # If string is empty or only whitespace
        with pytest.raises(ValueError, match="expr cannot be an empty string"):
            eval(s)

# Run hypothesis test manually for a few examples
test_inputs = ["", " ", "  ", "\n", "\t", "   \n   "]
for test_input in test_inputs:
    print(f"\nHypothesis test with input: {repr(test_input)}")
    try:
        if not test_input.strip():
            try:
                result = eval(test_input)
                print(f"  BUG: Expected ValueError but got result: {repr(result)}")
            except ValueError as e:
                if "expr cannot be an empty string" in str(e):
                    print(f"  PASS: Got expected ValueError")
                else:
                    print(f"  FAIL: Got ValueError but wrong message: {e}")
            except Exception as e:
                print(f"  FAIL: Got unexpected exception: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"  ERROR in test: {e}")

print("\n" + "=" * 60)
print("Testing edge cases")
print("=" * 60)

# Test valid expressions for comparison
valid_cases = [
    ("'1'", "1"),
    ("'1 + 1'", "1 + 1"),
    ("' 1 + 1 '", " 1 + 1 "),  # Leading/trailing spaces with valid expr
]

for display, test_str in valid_cases:
    print(f"\nTest: eval({display})")
    try:
        result = eval(test_str)
        print(f"Result: {repr(result)}")
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")