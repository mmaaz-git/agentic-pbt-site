#!/usr/bin/env python3

import sys
import os
# Add the starlette environment to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from starlette.convertors import FloatConvertor
import re

print("Testing FloatConvertor Round-Trip Property")
print("=" * 50)

# First test: Property-based test
print("\n1. Property-based test:")
@given(st.from_regex(re.compile(r"[0-9]+(\.[0-9]+)?"), fullmatch=True))
@settings(max_examples=100)
def test_float_convertor_round_trip(string_value):
    convertor = FloatConvertor()
    float_value = convertor.convert(string_value)
    reconstructed = convertor.to_string(float_value)
    original_float = float(string_value)
    try:
        assert convertor.convert(reconstructed) == original_float
    except AssertionError:
        print(f"  FAILURE: '{string_value}' -> {float_value} -> '{reconstructed}'")
        print(f"    Original float: {original_float}")
        print(f"    Reconstructed float: {convertor.convert(reconstructed)}")
        raise

try:
    test_float_convertor_round_trip()
    print("  Property-based test PASSED (100 examples)")
except Exception as e:
    print(f"  Property-based test FAILED")

# Second test: Specific failing case from bug report
print("\n2. Specific test case from bug report:")
convertor = FloatConvertor()

original = "0.000000000000000000001"
value = convertor.convert(original)
result = convertor.to_string(value)

print(f"  Input:  '{original}'")
print(f"  Float:  {value}")
print(f"  Output: '{result}'")

try:
    assert original == result
    print("  PASSED: Strings match")
except AssertionError:
    print(f"  FAILED: Expected '{original}', got '{result}'")

# Also check if the round-trip preserves the value
original_float = float(original)
reconstructed_float = convertor.convert(result)
print(f"\n  Original float value:     {original_float}")
print(f"  Reconstructed float value: {reconstructed_float}")

if original_float == reconstructed_float:
    print("  Float values match (round-trip successful)")
else:
    print("  Float values DO NOT match (round-trip failed)")

# Additional test cases
print("\n3. Additional edge cases:")
test_cases = [
    "0.1",
    "0.01",
    "0.001",
    "0.0001",
    "0.00001",
    "0.000001",
    "0.0000001",
    "0.00000001",
    "0.000000001",
    "0.0000000001",
    "0.00000000001",
    "0.000000000001",
    "0.0000000000001",
    "0.00000000000001",
    "0.000000000000001",
    "0.0000000000000001",
    "0.00000000000000001",
    "0.000000000000000001",
    "0.0000000000000000001",
    "0.00000000000000000001",  # 1e-20
    "0.000000000000000000001", # 1e-21
    "0.0000000000000000000001", # 1e-22
]

failures = []
for test_val in test_cases:
    val = convertor.convert(test_val)
    result = convertor.to_string(val)
    if float(test_val) != float(convertor.convert(result)):
        failures.append((test_val, val, result))
        print(f"  FAIL: '{test_val}' -> {val} -> '{result}'")

if not failures:
    print("  All additional test cases passed")
else:
    print(f"\n  Failed {len(failures)} out of {len(test_cases)} test cases")
    print(f"  Failures start at: {failures[0][0]} (value = {failures[0][1]})")