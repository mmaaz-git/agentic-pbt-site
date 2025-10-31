#!/usr/bin/env python3
"""Test script to reproduce the ensure_decoded bug"""

# First, test the hypothesis test
from hypothesis import given, strategies as st
from pandas.core.computation.common import ensure_decoded

@given(st.binary())
def test_ensure_decoded_returns_str(data):
    result = ensure_decoded(data)
    assert isinstance(result, str)

print("Running hypothesis test...")
try:
    test_ensure_decoded_returns_str()
    print("Hypothesis test passed (shouldn't happen if bug exists)")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

# Now test the specific failing case
print("\nTesting specific failing input b'\\x80'...")
data = b'\x80'
try:
    result = ensure_decoded(data)
    print(f"Result: {result!r}")
    print(f"Result type: {type(result)}")
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}")
except Exception as e:
    print(f"Other error: {e}")