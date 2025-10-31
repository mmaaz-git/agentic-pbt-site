#!/usr/bin/env python3
"""Hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
import pandas as pd
import re
import pytest

@given(st.lists(st.text(), min_size=1), st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789', min_size=1), st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789'))
@settings(max_examples=500)
def test_replace_removes_all_occurrences(strings, old, new):
    s = pd.Series(strings)
    count_before = s.str.count(old)
    replaced = s.str.replace(old, new, regex=False)
    count_after = replaced.str.count(old)

    for i in range(len(s)):
        if pd.notna(count_before.iloc[i]) and pd.notna(count_after.iloc[i]):
            assert count_after.iloc[i] == 0

# Also test with some specific examples that include regex metacharacters
def test_specific_metacharacters():
    """Test with specific regex metacharacters to show the issue"""
    print("\nTesting specific metacharacters:")

    # Test with parentheses
    s = pd.Series(['test(abc)test'])
    try:
        count = s.str.count('(')
        print(f"  Counting '(' in 'test(abc)test': {count.iloc[0]}")
    except Exception as e:
        print(f"  ERROR counting '(': {type(e).__name__}: {e}")

    # Test with dots
    s = pd.Series(['hello.world'])
    try:
        count = s.str.count('.')
        print(f"  Counting '.' in 'hello.world': {count.iloc[0]} (Expected: 1, Got: {count.iloc[0]})")
    except Exception as e:
        print(f"  ERROR counting '.': {type(e).__name__}: {e}")

if __name__ == "__main__":
    # Run the specific test
    test_specific_metacharacters()

    # Run hypothesis test with a few examples
    print("\nRunning hypothesis test with limited examples:")
    test_replace_removes_all_occurrences()
    print("Hypothesis test passed with alphanumeric strings!")

    # Now try with metacharacters
    print("\nTrying with metacharacters:")
    s = pd.Series(['test)test', 'hello(world'])
    try:
        count_before = s.str.count(')')
        print(f"  Count of ')': {count_before.tolist()}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        print("  The hypothesis test would fail with metacharacters!")