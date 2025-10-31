import pandas as pd
from pandas import DataFrame
from unittest.mock import patch

# Test 1: Reproduction test from bug report
print("=== Test 1: Basic Reproduction ===")
df = DataFrame({'a': [1, 2, 3]})

# Test read_clipboard with non-UTF-8 encoding
with patch('pandas.io.clipboard.clipboard_get', return_value='a\n1\n2\n3'):
    try:
        pd.read_clipboard(encoding='ascii')
    except Exception as e:
        print(f"read_clipboard with 'ascii': {type(e).__name__} - {str(e)}")

# Test to_clipboard with non-UTF-8 encoding
try:
    df.to_clipboard(encoding='ascii')
except Exception as e:
    print(f"to_clipboard with 'ascii': {type(e).__name__} - {str(e)}")

print("\n=== Test 2: Multiple encodings ===")
test_encodings = ['ascii', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']

for encoding in test_encodings:
    read_exception = None
    to_exception = None

    with patch('pandas.io.clipboard.clipboard_get', return_value='a\n1\n2\n3'):
        try:
            pd.read_clipboard(encoding=encoding)
        except Exception as e:
            read_exception = type(e).__name__

    try:
        df.to_clipboard(encoding=encoding)
    except Exception as e:
        to_exception = type(e).__name__

    print(f"Encoding '{encoding}': read_clipboard={read_exception}, to_clipboard={to_exception}")

print("\n=== Test 3: UTF-8 variants (should work) ===")
utf8_variants = ['utf-8', 'UTF-8', 'utf8', 'UTF8']

for encoding in utf8_variants:
    read_exception = None
    to_exception = None

    with patch('pandas.io.clipboard.clipboard_get', return_value='a\n1\n2\n3'):
        try:
            result = pd.read_clipboard(encoding=encoding)
            print(f"read_clipboard with '{encoding}': Success")
        except Exception as e:
            print(f"read_clipboard with '{encoding}': {type(e).__name__}")

    try:
        # We need to mock clipboard_set to avoid actual clipboard operations
        with patch('pandas.io.clipboard.clipboard_set'):
            df.to_clipboard(encoding=encoding)
            print(f"to_clipboard with '{encoding}': Success")
    except Exception as e:
        print(f"to_clipboard with '{encoding}': {type(e).__name__}")