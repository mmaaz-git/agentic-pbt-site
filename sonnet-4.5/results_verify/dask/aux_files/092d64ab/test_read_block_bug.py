#!/usr/bin/env python3
"""Test to reproduce the read_block bug"""

from io import BytesIO
from hypothesis import given, strategies as st, assume
from dask.bytes.core import read_block

# Test 1: Run the hypothesis test from the bug report
print("Running hypothesis test...")
@given(st.binary(min_size=1, max_size=1000), st.integers(min_value=0, max_value=100))
def test_read_block_length_none_reads_to_end(data, offset):
    assume(offset < len(data))
    f = BytesIO(data)
    result = read_block(f, offset, None, delimiter=None)
    expected = data[offset:]
    assert result == expected

try:
    test_read_block_length_none_reads_to_end()
    print("Hypothesis test passed (unexpected!)")
except AssertionError as e:
    print(f"Hypothesis test failed with AssertionError: {e}")
except Exception as e:
    print(f"Hypothesis test failed with other error: {e}")

# Test 2: Simple reproduction from bug report
print("\nRunning simple reproduction...")
try:
    data = b"Hello World!"
    f = BytesIO(data)
    result = read_block(f, 0, None, delimiter=None)
    print(f"Result: {result}")
except AssertionError as e:
    print(f"AssertionError raised (as expected): {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test with delimiter (should work according to bug report)
print("\nTesting with delimiter...")
try:
    data = b"Hello\nWorld!\nTest"
    f = BytesIO(data)
    result = read_block(f, 0, None, delimiter=b'\n')
    print(f"Result with delimiter: {result}")
    print("Works with delimiter!")
except AssertionError as e:
    print(f"AssertionError with delimiter: {e}")
except Exception as e:
    print(f"Other error with delimiter: {e}")

# Test 4: Test with normal length parameter (should work)
print("\nTesting with normal length parameter...")
try:
    data = b"Hello World!"
    f = BytesIO(data)
    result = read_block(f, 0, 5, delimiter=None)
    print(f"Result with length=5: {result}")
    assert result == b"Hello"
    print("Works with normal length!")
except Exception as e:
    print(f"Error with normal length: {e}")