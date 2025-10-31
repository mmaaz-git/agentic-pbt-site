#!/usr/bin/env python3
"""Test script to reproduce the reported bug"""

from unittest.mock import patch
import pandas as pd
import re
import traceback

def test_separator(sep, test_data=None):
    """Test a specific separator"""
    if test_data is None:
        test_data = f"a{sep}b\n1{sep}2\n3{sep}4"

    print(f"\nTesting separator: {repr(sep)}")
    print(f"Test data: {repr(test_data[:50])}...")

    with patch('pandas.io.clipboard.clipboard_get', return_value=test_data):
        try:
            result = pd.read_clipboard(sep=sep)
            print(f"  Success! Result shape: {result.shape}")
            print(f"  Columns: {list(result.columns)}")
            return True, None
        except Exception as e:
            print(f"  Failed with {type(e).__name__}: {e}")
            return False, e

# Test the specific examples from the bug report
print("=" * 60)
print("Testing specific examples from bug report:")
print("=" * 60)

# These should fail according to the bug report
failing_seps = ['**', '++', '0(', '0)', '[[']
for sep in failing_seps:
    success, error = test_separator(sep)
    if not success and isinstance(error, re.error):
        print(f"  ✓ Confirmed: regex error as reported")

print("\n" + "=" * 60)
print("Testing separators that should work:")
print("=" * 60)

# These should work according to the bug report
working_seps = ['::', '||', 'ab', '123']
for sep in working_seps:
    test_separator(sep)

print("\n" + "=" * 60)
print("Testing the Hypothesis property test:")
print("=" * 60)

from hypothesis import given, assume, strategies as st
import re

@given(st.text(min_size=2, max_size=5))
def test_multi_char_sep_uses_python_engine(sep):
    """Multi-character separators should either work or give helpful errors"""
    from pandas.io.clipboards import read_clipboard

    assume(len(sep) > 1)

    test_data = f"a{sep}b\n1{sep}2\n3{sep}4"

    with patch('pandas.io.clipboard.clipboard_get', return_value=test_data):
        try:
            result = read_clipboard(sep=sep)
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # Should not crash with low-level regex errors
            if isinstance(e, re.error):
                print(f"  Found regex error with sep={repr(sep)}: {e}")
                return False
    return True

# Run a limited test
print("\nRunning hypothesis test with some examples...")
test_cases = ['**', '++', '((', '))', '[[', ']]', '??', '..', '\\\\']
failures = []
for sep in test_cases:
    if len(sep) > 1:
        try:
            if not test_multi_char_sep_uses_python_engine(sep):
                failures.append(sep)
        except:
            pass

if failures:
    print(f"\nSeparators that caused regex errors: {failures}")

print("\n" + "=" * 60)
print("Testing if re.escape() workaround works:")
print("=" * 60)

# Test if escaping helps
for sep in ['**', '++', '0(']:
    escaped_sep = re.escape(sep)
    print(f"\nOriginal sep: {repr(sep)}, Escaped: {repr(escaped_sep)}")
    success, error = test_separator(escaped_sep, test_data=f"a{sep}b\n1{sep}2\n3{sep}4")