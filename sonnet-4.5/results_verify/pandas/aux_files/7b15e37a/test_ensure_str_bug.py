#!/usr/bin/env python3
"""Test the ensure_str bug report"""

from hypothesis import given, settings, strategies as st
from pandas.core.dtypes.common import ensure_str
import traceback

# First, let's test with the Hypothesis test from the bug report
@settings(max_examples=1000)
@given(
    st.one_of(
        st.binary(),
        st.text(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
    )
)
def test_ensure_str_returns_str(value):
    result = ensure_str(value)
    assert isinstance(result, str), f"Expected str, got {type(result)}"

print("Running Hypothesis test...")
try:
    test_ensure_str_returns_str()
    print("Hypothesis test passed")
except Exception as e:
    print(f"Hypothesis test failed: {e}")
    traceback.print_exc()

# Now test the specific failing input
print("\nTesting specific failing input b'\\x80'...")
invalid_utf8_bytes = b'\x80'
try:
    result = ensure_str(invalid_utf8_bytes)
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError occurred: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Other exception occurred: {e}")
    traceback.print_exc()

# Test some other cases
print("\nTesting valid UTF-8 bytes...")
valid_utf8 = b'hello'
try:
    result = ensure_str(valid_utf8)
    print(f"Result for valid UTF-8: '{result}', type: {type(result)}")
except Exception as e:
    print(f"Exception on valid UTF-8: {e}")

print("\nTesting string input...")
try:
    result = ensure_str("already a string")
    print(f"Result for string: '{result}', type: {type(result)}")
except Exception as e:
    print(f"Exception on string: {e}")

print("\nTesting integer input...")
try:
    result = ensure_str(42)
    print(f"Result for integer: '{result}', type: {type(result)}")
except Exception as e:
    print(f"Exception on integer: {e}")