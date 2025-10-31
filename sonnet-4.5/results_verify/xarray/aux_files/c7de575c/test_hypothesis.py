#!/usr/bin/env python3
"""Property-based test for xarray string encoding/decoding"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st, settings, example
import xarray.coding.strings as xr_strings

@given(st.text(min_size=1, max_size=100))
@settings(max_examples=1000)
@example('\x00')  # Specific example mentioned in bug report
@example('hello\x00world')  # String with null in middle
@example('\x00\x00\x00')  # Multiple nulls
def test_encode_decode_string_roundtrip(text):
    arr = np.array([text], dtype=object)

    encoded = xr_strings.encode_string_array(arr, encoding='utf-8')
    decoded = xr_strings.decode_bytes_array(encoded, encoding='utf-8')

    try:
        np.testing.assert_array_equal(decoded, arr)
        print(f"✓ Pass: {repr(text[:20])}" + ("..." if len(text) > 20 else ""))
    except AssertionError as e:
        print(f"✗ FAIL: {repr(text)}")
        print(f"  Original: {repr(arr[0])}, length: {len(arr[0])}")
        print(f"  Decoded:  {repr(decoded[0])}, length: {len(decoded[0])}")
        print(f"  Encoded dtype: {encoded.dtype}")
        print(f"  Encoded value: {repr(encoded[0])}")
        raise

if __name__ == "__main__":
    print("Running property-based test with Hypothesis...")
    test_encode_decode_string_roundtrip()
    print("\nAll tests passed!")