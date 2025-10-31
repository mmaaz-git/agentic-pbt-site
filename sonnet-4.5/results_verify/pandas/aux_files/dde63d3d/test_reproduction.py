#!/usr/bin/env python3
"""Test script to reproduce the bug in pandas.io.clipboard"""

import pandas as pd
from pandas import read_clipboard  # read_clipboard is in pandas namespace
# to_clipboard is a DataFrame method

print("Testing exception type inconsistency in pandas clipboard functions:")
print("=" * 60)

# Test read_clipboard with non-UTF8 encoding
print("\n1. Testing read_clipboard with 'latin-1' encoding:")
try:
    read_clipboard(encoding='latin-1')
except Exception as e:
    print(f"   Exception type: {type(e).__name__}")
    print(f"   Exception message: {e}")

# Test to_clipboard with non-UTF8 encoding
print("\n2. Testing to_clipboard with 'latin-1' encoding:")
try:
    df = pd.DataFrame([[1, 2]])
    df.to_clipboard(encoding='latin-1')
except Exception as e:
    print(f"   Exception type: {type(e).__name__}")
    print(f"   Exception message: {e}")

# Test multiple encodings
print("\n3. Testing multiple non-UTF8 encodings:")
encodings_to_test = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'ascii']

for encoding in encodings_to_test:
    read_exc = None
    write_exc = None

    try:
        read_clipboard(encoding=encoding)
    except Exception as e:
        read_exc = type(e).__name__

    try:
        pd.DataFrame([[1]]).to_clipboard(encoding=encoding)
    except Exception as e:
        write_exc = type(e).__name__

    print(f"   {encoding:12} - read_clipboard: {read_exc:20} to_clipboard: {write_exc}")

print("\n4. Testing that UTF-8 encoding is allowed:")
# Test that utf-8 variations work (or at least don't raise encoding errors)
utf8_variants = ['utf-8', 'UTF-8', 'utf8', 'UTF8']
for enc in utf8_variants:
    try:
        # We expect these might fail for clipboard access reasons,
        # but not for encoding validation
        read_clipboard(encoding=enc)
    except NotImplementedError as e:
        if "utf-8 encoding" in str(e).lower():
            print(f"   {enc} incorrectly rejected in read_clipboard")
    except ValueError as e:
        if "utf-8 encoding" in str(e).lower():
            print(f"   {enc} incorrectly rejected in read_clipboard")
    except Exception:
        # Other exceptions (like clipboard access) are expected
        pass

print("\nReproduction complete!")