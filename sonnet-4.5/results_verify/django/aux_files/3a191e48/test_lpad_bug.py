"""Test script to reproduce the _sqlite_lpad bug"""
import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.backends.sqlite3._functions import _sqlite_lpad
from hypothesis import given, strategies as st, settings

# Hypothesis test
@given(st.text(), st.integers(min_value=0, max_value=10000), st.text(min_size=0))
@settings(max_examples=10000)
def test_lpad_length_invariant(text, length, fill_text):
    result = _sqlite_lpad(text, length, fill_text)
    if result is not None:
        assert len(result) == length, f"Expected length {length}, but got {len(result)} for _sqlite_lpad('{text}', {length}, '{fill_text}')"

# Direct reproduction tests
def test_specific_failing_cases():
    print("Testing specific failing cases:")
    print("=" * 50)

    # Test case 1: Empty strings
    result1 = _sqlite_lpad('', 1, '')
    print(f"_sqlite_lpad('', 1, '')")
    print(f"  Expected length: 1")
    print(f"  Actual result: '{result1}'")
    print(f"  Actual length: {len(result1)}")
    print(f"  PASS: {len(result1) == 1}")
    print()

    # Test case 2: hello with empty fill
    result2 = _sqlite_lpad('hello', 10, '')
    print(f"_sqlite_lpad('hello', 10, '')")
    print(f"  Expected length: 10")
    print(f"  Actual result: '{result2}'")
    print(f"  Actual length: {len(result2)}")
    print(f"  PASS: {len(result2) == 10}")
    print()

    # Test case 3: x with empty fill
    result3 = _sqlite_lpad('x', 5, '')
    print(f"_sqlite_lpad('x', 5, '')")
    print(f"  Expected length: 5")
    print(f"  Actual result: '{result3}'")
    print(f"  Actual length: {len(result3)}")
    print(f"  PASS: {len(result3) == 5}")
    print()

    # Test normal case for comparison
    result4 = _sqlite_lpad('hello', 10, '*')
    print(f"_sqlite_lpad('hello', 10, '*')")
    print(f"  Expected length: 10")
    print(f"  Actual result: '{result4}'")
    print(f"  Actual length: {len(result4)}")
    print(f"  PASS: {len(result4) == 10}")
    print()

if __name__ == "__main__":
    # First run specific test cases
    test_specific_failing_cases()

    print("=" * 50)
    print("Running Hypothesis tests...")
    print("=" * 50)

    # Run hypothesis test
    try:
        test_lpad_length_invariant()
        print("Hypothesis test passed!")
    except AssertionError as e:
        print(f"Hypothesis test failed: {e}")