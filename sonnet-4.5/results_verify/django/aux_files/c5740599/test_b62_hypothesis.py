#!/usr/bin/env python3
"""Proper hypothesis test for b62_decode bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings, assume
from django.core.signing import b62_encode, b62_decode

# Test the property-based test from the bug report
@given(st.text(min_size=1).filter(lambda x: all(c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-' for c in x)))
@settings(max_examples=100)
def test_b62_decode_encode_roundtrip(s):
    """Test that decode-encode is a round-trip for valid base62 strings"""
    decoded = b62_decode(s)
    re_encoded = b62_encode(decoded)
    assert re_encoded == s, f"Decode-encode round-trip failed: '{s}' -> {decoded} -> '{re_encoded}'"

if __name__ == "__main__":
    print("Running hypothesis test for round-trip property...")
    try:
        test_b62_decode_encode_roundtrip()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Assertion failed: {e}")
    except Exception as e:
        print(f"Test failed with exception: {e}")