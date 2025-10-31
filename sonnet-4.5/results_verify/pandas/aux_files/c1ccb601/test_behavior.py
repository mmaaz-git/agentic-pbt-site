#!/usr/bin/env python3
"""Test to understand the exact behavior"""

from unittest.mock import patch
import pandas as pd
import re

print("Testing what the actual behavior is:")
print("=" * 60)

# Test 1: What happens with single character regex metacharacters?
print("\n1. Single character regex metacharacters:")
for sep in ['*', '+', '(', ')', '[', ']', '.', '?']:
    test_data = f"a{sep}b\n1{sep}2"
    with patch('pandas.io.clipboard.clipboard_get', return_value=test_data):
        try:
            result = pd.read_clipboard(sep=sep)
            print(f"  sep={repr(sep)}: Works (shape={result.shape})")
        except Exception as e:
            print(f"  sep={repr(sep)}: {type(e).__name__}: {e}")

# Test 2: Check if documentation is clear in read_csv
print("\n2. Let's test read_csv directly with the same separators:")
from io import StringIO

for sep in ['**', '++', '((']:
    test_data = f"a{sep}b\n1{sep}2"
    try:
        result = pd.read_csv(StringIO(test_data), sep=sep)
        print(f"  sep={repr(sep)}: Works (shape={result.shape})")
    except Exception as e:
        print(f"  sep={repr(sep)}: {type(e).__name__}: {e}")

# Test 3: Check engines
print("\n3. Testing with explicit engine choices:")
test_data = "a**b\n1**2"

for engine in ['c', 'python', None]:
    with patch('pandas.io.clipboard.clipboard_get', return_value=test_data):
        print(f"\n  Engine={engine}:")
        try:
            result = pd.read_clipboard(sep='**', engine=engine)
            print(f"    Works! Shape: {result.shape}")
        except Exception as e:
            print(f"    {type(e).__name__}: {e}")

# Test 4: Check if '||' truly works as expected
print("\n4. Testing '||' separator more carefully:")
test_data = "a||b||c\n1||2||3"
with patch('pandas.io.clipboard.clipboard_get', return_value=test_data):
    result = pd.read_clipboard(sep='||')
    print(f"  Columns: {list(result.columns)}")
    print(f"  Shape: {result.shape}")
    print(f"  Note: '||' is being treated as regex '|' OR '|' which matches every position!")

# Test 5: Valid regex patterns that work correctly
print("\n5. Testing valid regex patterns:")
for sep, test_data in [
    (r'\s+', "a  b\n1   2"),
    (r'\t+', "a\t\tb\n1\t2"),
    (r'[,;]', "a,b;c\n1;2,3"),
]:
    with patch('pandas.io.clipboard.clipboard_get', return_value=test_data):
        try:
            result = pd.read_clipboard(sep=sep)
            print(f"  sep={repr(sep)}: Works (shape={result.shape})")
        except Exception as e:
            print(f"  sep={repr(sep)}: {type(e).__name__}: {e}")