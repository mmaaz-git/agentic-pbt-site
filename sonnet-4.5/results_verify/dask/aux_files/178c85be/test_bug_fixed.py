#!/usr/bin/env python3
"""Test the reported bug in dask.widgets.widgets.key_split"""

from hypothesis import given, strategies as st, settings
from dask.widgets.widgets import key_split

# First test the regular case that should work
print("Testing regular case with valid UTF-8 bytes:")
try:
    result = key_split(b'hello-world-1')
    print(f"key_split(b'hello-world-1') = '{result}'")
    print("✓ Valid UTF-8 bytes work correctly")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

print("\n" + "="*50 + "\n")

# Now test the failing case with invalid UTF-8
print("Testing invalid UTF-8 bytes (b'\\x80'):")
try:
    result = key_split(b'\x80')
    print(f"key_split(b'\\x80') = '{result}'")
    print("✓ Invalid UTF-8 bytes handled gracefully")
except UnicodeDecodeError as e:
    print(f"✗ UnicodeDecodeError raised: {e}")
    print(f"Error type: {type(e).__name__}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

print("\n" + "="*50 + "\n")

# Test with hypothesis
print("Running hypothesis property test:")
failures = []

@given(st.binary(min_size=0, max_size=100))
@settings(max_examples=100)
def test_key_split_bytes(b):
    try:
        result = key_split(b)
        assert isinstance(result, str), f"Result should be string, got {type(result)}"
    except Exception as e:
        failures.append((b, e))
        raise

# Run the test
try:
    test_key_split_bytes()
    print("✓ Hypothesis test passed for 100 examples")
except Exception as e:
    print(f"✗ Hypothesis test failed")
    if failures:
        print(f"First failure: bytes={failures[0][0]!r}")
        print(f"Error: {failures[0][1]}")

print("\n" + "="*50 + "\n")

# Test that None returns "Other" as documented
print("Testing None input:")
try:
    result = key_split(None)
    print(f"key_split(None) = '{result}'")
    if result == "Other":
        print("✓ None correctly returns 'Other'")
    else:
        print(f"✗ Expected 'Other', got '{result}'")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

print("\n" + "="*50 + "\n")

# Test various invalid UTF-8 sequences
print("Testing various invalid UTF-8 sequences:")
invalid_sequences = [
    b'\x80',  # Invalid start byte
    b'\xff',  # Invalid byte
    b'\xc0\x80',  # Overlong encoding
    b'\xed\xa0\x80',  # UTF-16 surrogate half
    b'\xc2',  # Incomplete sequence
]

for seq in invalid_sequences:
    try:
        result = key_split(seq)
        print(f"key_split({seq!r}) = '{result}' ✓")
    except UnicodeDecodeError as e:
        print(f"key_split({seq!r}) raised UnicodeDecodeError ✗")