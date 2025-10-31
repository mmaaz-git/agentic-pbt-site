#!/usr/bin/env python3
"""Test script to reproduce the b62_decode bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings
from django.core.signing import b62_encode, b62_decode

# Test the property-based test from the bug report
@given(st.text(min_size=1).filter(lambda x: all(c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-' for c in x)))
@settings(max_examples=500)
def test_b62_decode_encode_roundtrip(s):
    try:
        decoded = b62_decode(s)
        re_encoded = b62_encode(decoded)
        if re_encoded != s:
            print(f"Round-trip failed: '{s}' -> {decoded} -> '{re_encoded}'")
            return False
    except Exception as e:
        print(f"Exception for input '{s}': {e}")
        return False
    return True

# Specific test cases mentioned in the bug report
def test_specific_cases():
    print("Testing specific cases:")
    print("-" * 50)

    # Test '-'
    try:
        result = b62_decode('-')
        print(f"b62_decode('-') = {result}")
        print(f"b62_encode({result}) = '{b62_encode(result)}'")
        print(f"Round-trip successful: {b62_encode(b62_decode('-')) == '-'}")
    except Exception as e:
        print(f"Exception for '-': {e}")

    print()

    # Test '-0'
    try:
        result = b62_decode('-0')
        print(f"b62_decode('-0') = {result}")
        print(f"b62_encode({result}) = '{b62_encode(result)}'")
        print(f"Round-trip successful: {b62_encode(b62_decode('-0')) == '-0'}")
    except Exception as e:
        print(f"Exception for '-0': {e}")

    print()

    # Test '0'
    try:
        result = b62_decode('0')
        print(f"b62_decode('0') = {result}")
        print(f"b62_encode({result}) = '{b62_encode(result)}'")
        print(f"Round-trip successful: {b62_encode(b62_decode('0')) == '0'}")
    except Exception as e:
        print(f"Exception for '0': {e}")

    print()

    # Test valid negative number
    try:
        result = b62_decode('-5')
        print(f"b62_decode('-5') = {result}")
        print(f"b62_encode({result}) = '{b62_encode(result)}'")
        print(f"Round-trip successful: {b62_encode(b62_decode('-5')) == '-5'}")
    except Exception as e:
        print(f"Exception for '-5': {e}")

# Test what b62_encode produces for negative zero
def test_encode_negative_zero():
    print("\nTesting b62_encode for 0 and -0:")
    print("-" * 50)
    print(f"b62_encode(0) = '{b62_encode(0)}'")
    print(f"b62_encode(-0) = '{b62_encode(-0)}'")  # In Python, -0 == 0

if __name__ == "__main__":
    print("Running specific test cases from bug report:")
    print("=" * 60)
    test_specific_cases()

    test_encode_negative_zero()

    print("\nRunning property-based tests:")
    print("=" * 60)

    # Run the hypothesis test
    try:
        test_b62_decode_encode_roundtrip()
        print("Property-based test completed")
    except Exception as e:
        print(f"Property-based test failed: {e}")