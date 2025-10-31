#!/usr/bin/env python3
"""Test script to reproduce the reported bug in dask.utils.key_split"""

import sys
import traceback

# First test: The property-based test
print("=" * 60)
print("Test 1: Property-based test with hypothesis")
print("=" * 60)

try:
    from hypothesis import given, strategies as st
    from dask.widgets import FILTERS

    @given(st.binary(min_size=1))
    def test_key_split_bytes_returns_string(b):
        key_split = FILTERS['key_split']
        result = key_split(b)
        assert isinstance(result, str)

    # Run the test
    test_key_split_bytes_returns_string()
    print("Property-based test completed (no error raised)")

except Exception as e:
    print(f"Property-based test failed with error:")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {str(e)}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test 2: Direct reproduction with b'\\x80'")
print("=" * 60)

# Second test: Direct reproduction
try:
    from dask.utils import key_split

    print(f"Calling key_split(b'\\x80')...")
    result = key_split(b'\x80')
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")

except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError raised as reported:")
    print(f"Message: {str(e)}")

except Exception as e:
    print(f"Different error raised:")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {str(e)}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test 3: Test with valid UTF-8 bytes (from docstring)")
print("=" * 60)

# Third test: Test the example from docstring
try:
    from dask.utils import key_split

    print(f"Calling key_split(b'hello-world-1')...")
    result = key_split(b'hello-world-1')
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")

except Exception as e:
    print(f"Error raised:")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {str(e)}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test 4: Check what FILTERS['key_split'] actually is")
print("=" * 60)

try:
    from dask.widgets import FILTERS
    from dask.utils import key_split as utils_key_split

    widget_key_split = FILTERS['key_split']
    print(f"FILTERS['key_split'] is: {widget_key_split}")
    print(f"dask.utils.key_split is: {utils_key_split}")
    print(f"Are they the same? {widget_key_split is utils_key_split}")

except Exception as e:
    print(f"Error: {e}")