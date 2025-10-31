#!/usr/bin/env python3
"""Test the reported FloatConvertor bug"""

from hypothesis import given, strategies as st, settings
from starlette.convertors import FloatConvertor
import re

# First, let's reproduce the exact failing case
def test_specific_case():
    print("Testing specific case: 0.000000000000000000001")
    convertor = FloatConvertor()

    original = "0.000000000000000000001"
    value = convertor.convert(original)
    result = convertor.to_string(value)

    print(f"Input:  '{original}'")
    print(f"Float:  {value}")
    print(f"Output: '{result}'")
    print(f"Float value in scientific notation: {value:.30e}")
    print(f"Original as float: {float(original)}")
    print(f"Result as float: {float(result) if result and result != '.' else 0.0}")

    # Check the assertion
    try:
        assert float(result) == float(original)
        print("ASSERTION PASSED")
    except AssertionError as e:
        print(f"ASSERTION FAILED: {float(result)} != {float(original)}")
    print()

# Test with property-based testing
@given(st.from_regex(re.compile(r"[0-9]+(\.[0-9]+)?"), fullmatch=True))
@settings(max_examples=100)
def test_float_convertor_round_trip(string_value):
    convertor = FloatConvertor()
    try:
        float_value = convertor.convert(string_value)
        reconstructed = convertor.to_string(float_value)
        original_float = float(string_value)
        assert convertor.convert(reconstructed) == original_float
    except AssertionError:
        print(f"Failed on: {string_value}")
        print(f"  Original float: {float(string_value)}")
        print(f"  Converted: {float_value}")
        print(f"  Reconstructed string: {reconstructed}")
        print(f"  Reconstructed float: {float(reconstructed) if reconstructed and reconstructed != '.' else 0.0}")
        raise

# Test edge cases
def test_edge_cases():
    print("Testing various edge cases:")
    convertor = FloatConvertor()

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
        "0.00000000000000000001",  # 20 decimal places
        "0.000000000000000000001", # 21 decimal places - this should fail
        "0.0000000000000000000001", # 22 decimal places
        "1.123456789012345678901234567890", # Many decimal places
        "999999999999999999999.999999999999999999999999"
    ]

    for test_val in test_cases:
        try:
            float_val = convertor.convert(test_val)
            reconstructed = convertor.to_string(float_val)
            print(f"  {test_val[:30]:30s} -> {reconstructed:20s} | Match: {float(reconstructed) == float(test_val)}")
        except Exception as e:
            print(f"  {test_val[:30]:30s} -> ERROR: {e}")
    print()

# Test the formatting behavior
def test_formatting():
    print("Testing formatting behavior of to_string:")
    convertor = FloatConvertor()

    # Test how Python handles very small numbers
    small_num = 0.000000000000000000001  # 1e-21
    print(f"Python float representation: {small_num}")
    print(f"Scientific notation: {small_num:.30e}")
    print(f"Using %0.20f format: '%0.20f' % {small_num} = '{('%0.20f' % small_num)}'")
    print(f"After stripping: '{('%0.20f' % small_num).rstrip('0').rstrip('.')}'")
    print()

    # Test the actual convertor
    result = convertor.to_string(small_num)
    print(f"convertor.to_string({small_num}) = '{result}'")
    print()

if __name__ == "__main__":
    test_specific_case()
    test_edge_cases()
    test_formatting()

    print("Running property-based tests...")
    try:
        test_float_convertor_round_trip()
        print("Property-based tests passed!")
    except Exception as e:
        print(f"Property-based test failed: {e}")