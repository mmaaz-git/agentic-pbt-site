#!/usr/bin/env python3
"""Test script to reproduce the encoding validation bug."""

import pandas as pd
from pandas.io.clipboards import read_clipboard, to_clipboard
import traceback

def test_encoding_validation():
    """Test that various valid UTF-8 encoding names work."""

    # First, verify that Python accepts these encodings
    print("Testing Python's encoding support:")
    test_encodings = ["utf-8", "UTF-8", "utf_8", "UTF_8", "Utf-8", "utf8", "UTF8"]

    for encoding in test_encodings:
        try:
            text = "hello world"
            encoded = text.encode(encoding)
            print(f"  {encoding:10} - Python accepts this encoding: {encoded[:20]}...")
        except Exception as e:
            print(f"  {encoding:10} - Python REJECTED: {e}")

    print("\n" + "="*60)
    print("Testing pandas read_clipboard encoding validation:")
    print("="*60)

    for encoding in test_encodings:
        try:
            # Note: This will fail if clipboard is empty, but we're testing encoding validation
            # which happens before clipboard read
            result = read_clipboard(encoding=encoding)
            print(f"  {encoding:10} - ACCEPTED by read_clipboard")
        except NotImplementedError as e:
            if "only supports utf-8 encoding" in str(e):
                print(f"  {encoding:10} - REJECTED: {e}")
            else:
                print(f"  {encoding:10} - Other NotImplementedError: {e}")
        except Exception as e:
            # Clipboard might be empty or other issues - that's fine for our test
            print(f"  {encoding:10} - Other error (not encoding related): {type(e).__name__}")

    print("\n" + "="*60)
    print("Testing pandas to_clipboard encoding validation:")
    print("="*60)

    df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])

    for encoding in test_encodings:
        try:
            to_clipboard(df, encoding=encoding)
            print(f"  {encoding:10} - ACCEPTED by to_clipboard")
        except ValueError as e:
            if "clipboard only supports utf-8 encoding" in str(e):
                print(f"  {encoding:10} - REJECTED: {e}")
            else:
                print(f"  {encoding:10} - Other ValueError: {e}")
        except Exception as e:
            # Clipboard might not be available - that's fine for our test
            print(f"  {encoding:10} - Other error (not encoding related): {type(e).__name__}")

if __name__ == "__main__":
    test_encoding_validation()