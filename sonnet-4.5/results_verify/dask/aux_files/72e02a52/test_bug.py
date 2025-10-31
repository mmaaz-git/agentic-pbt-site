#!/usr/bin/env python3
"""Test the reported bug in dask.utils.key_split with non-UTF-8 bytes"""

from hypothesis import given, strategies as st, settings
from dask.utils import key_split
import traceback

# First, test the specific failing case mentioned in the bug report
print("Testing specific case: b'\\x80'")
try:
    result = key_split(b'\x80')
    print(f"Result: {result}")
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError caught: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Other exception caught: {e}")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test that UTF-8 bytes work as shown in docstring
print("Testing UTF-8 bytes from docstring example: b'hello-world-1'")
try:
    result = key_split(b'hello-world-1')
    print(f"Result: {result}")
    assert result == 'hello-world', f"Expected 'hello-world' but got '{result}'"
    print("✓ UTF-8 bytes work correctly")
except Exception as e:
    print(f"Unexpected error: {e}")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test other problematic inputs that return "Other"
print("Testing None input (should return 'Other'):")
try:
    result = key_split(None)
    print(f"Result: {result}")
    assert result == 'Other', f"Expected 'Other' but got '{result}'"
    print("✓ None input correctly returns 'Other'")
except Exception as e:
    print(f"Unexpected error: {e}")

print("\n" + "="*50 + "\n")

# Run the property-based test
print("Running property-based test with Hypothesis:")
failures = []

@given(st.binary())
@settings(max_examples=300)
def test_key_split_bytes(b):
    try:
        result = key_split(b)
        assert isinstance(result, str), f"key_split should return str for bytes, got {type(result)}"
    except UnicodeDecodeError as e:
        failures.append((b, e))
        raise

try:
    test_key_split_bytes()
    print("✓ All property-based tests passed!")
except Exception as e:
    print(f"Property-based test failed!")
    if failures:
        print(f"Found {len(failures)} failing inputs")
        print(f"First failure: {failures[0][0]!r}")
        print(f"Error: {failures[0][1]}")

print("\n" + "="*50 + "\n")

# Test more non-UTF-8 sequences
print("Testing various non-UTF-8 byte sequences:")
test_cases = [
    b'\x80',  # Invalid start byte
    b'\xff',  # Another invalid byte
    b'\xc0\x80',  # Overlong encoding
    b'\xed\xa0\x80',  # UTF-16 surrogate
    b'\xf5\x80\x80\x80',  # Out of range
    b'hello\x80world',  # Mixed valid/invalid
]

for test_bytes in test_cases:
    try:
        result = key_split(test_bytes)
        print(f"  {test_bytes!r:30} -> {result!r}")
    except UnicodeDecodeError as e:
        print(f"  {test_bytes!r:30} -> UnicodeDecodeError: {e}")
    except Exception as e:
        print(f"  {test_bytes!r:30} -> Exception: {e}")