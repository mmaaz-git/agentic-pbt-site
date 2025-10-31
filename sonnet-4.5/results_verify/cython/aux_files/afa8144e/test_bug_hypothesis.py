#!/usr/bin/env python3
"""Test the property-based test from the bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm
import struct
from hypothesis import given, strategies as st

@given(st.binary(min_size=1, max_size=100))
def test_decode_validates_input_length(binary_data):
    if len(binary_data) % 4 != 0:
        try:
            result = llm.decode(binary_data)
            assert False, f"decode should reject invalid input but accepted {len(binary_data)} bytes"
        except (ValueError, struct.error):
            pass
    else:
        result = llm.decode(binary_data)
        assert len(result) == len(binary_data) // 4

# Run the test with the specific failing input
print("Testing with specific failing input: b'\\x00\\x00\\x00\\x00\\xFF' (5 bytes)")
binary_data = b'\x00\x00\x00\x00\xFF'
print(f"Input length: {len(binary_data)} bytes")

try:
    result = llm.decode(binary_data)
    print(f"Result: {result}")
    print(f"Number of floats decoded: {len(result)}")
    print("ISSUE CONFIRMED: decode() accepted non-multiple-of-4 input and silently truncated")
except (ValueError, struct.error) as e:
    print(f"Exception raised (expected): {e}")

# Also run the general test
print("\nRunning property-based test...")
try:
    test_decode_validates_input_length()
    print("Property test passed")
except AssertionError as e:
    print(f"Property test failed with: {e}")