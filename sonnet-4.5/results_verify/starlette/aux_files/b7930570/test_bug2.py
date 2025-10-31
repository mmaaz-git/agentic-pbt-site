#!/usr/bin/env python3
"""Test the reported FloatConvertor bug with better property testing"""

from hypothesis import given, strategies as st, settings, example
from starlette.convertors import FloatConvertor
import re

# Test with property-based testing, catching failures
@given(st.from_regex(re.compile(r"[0-9]+(\.[0-9]+)?"), fullmatch=True))
@example("0.000000000000000000001")  # Force test the known failing case
@settings(max_examples=1000)
def test_float_convertor_round_trip(string_value):
    convertor = FloatConvertor()
    float_value = convertor.convert(string_value)
    reconstructed = convertor.to_string(float_value)
    original_float = float(string_value)

    # The assertion from the bug report
    if convertor.convert(reconstructed) != original_float:
        print(f"\nRound-trip failure found:")
        print(f"  Input string: '{string_value}'")
        print(f"  As float: {original_float} ({original_float:.30e})")
        print(f"  to_string result: '{reconstructed}'")
        print(f"  Final float: {convertor.convert(reconstructed)}")
        print(f"  Match: {convertor.convert(reconstructed) == original_float}")
        raise AssertionError(f"Round-trip failed for {string_value}")

if __name__ == "__main__":
    print("Running property-based test with explicit failing example...")
    try:
        test_float_convertor_round_trip()
        print("All tests passed!")
    except AssertionError as e:
        print(f"\nTest failed as expected: {e}")

    # Let's also check what kinds of strings the regex actually produces
    print("\n\nChecking what the regex can generate:")
    from hypothesis import strategies as st
    import re

    regex = re.compile(r"[0-9]+(\.[0-9]+)?")
    strategy = st.from_regex(regex, fullmatch=True)

    print("Sample values from regex strategy:")
    samples = []
    for _ in range(20):
        samples.extend(strategy.example())

    for sample in samples[:10]:
        if '.' in sample and len(sample) > 20:
            print(f"  Long decimal: {sample[:30]}... (len={len(sample)})")

    # Check if regex can generate arbitrarily long decimals
    print("\nTrying to generate very long decimals...")
    for i in range(100):
        sample = strategy.example()
        if '.' in sample:
            decimal_part = sample.split('.')[1]
            if len(decimal_part) > 20:
                print(f"  Found {len(decimal_part)}-digit decimal: {sample[:50]}...")
                break