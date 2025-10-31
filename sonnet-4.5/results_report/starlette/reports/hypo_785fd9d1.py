from hypothesis import given, strategies as st, settings
from starlette.convertors import FloatConvertor
import re

@given(st.from_regex(re.compile(r"[0-9]+(\.[0-9]+)?"), fullmatch=True))
@settings(max_examples=100)
def test_float_convertor_round_trip(string_value):
    convertor = FloatConvertor()
    float_value = convertor.convert(string_value)
    reconstructed = convertor.to_string(float_value)
    original_float = float(string_value)
    assert convertor.convert(reconstructed) == original_float, \
        f"Round-trip failed: '{string_value}' -> {float_value} -> '{reconstructed}' -> {convertor.convert(reconstructed)} != {original_float}"

if __name__ == "__main__":
    # Run the test
    try:
        test_float_convertor_round_trip()
        print("All property tests passed")
    except AssertionError as e:
        print(f"Property test failed: {e}")

    # Also test the specific failing case
    print("\nTesting specific failing case: '0.000000000000000000001'")
    try:
        convertor = FloatConvertor()
        string_value = "0.000000000000000000001"
        float_value = convertor.convert(string_value)
        reconstructed = convertor.to_string(float_value)
        original_float = float(string_value)
        assert convertor.convert(reconstructed) == original_float, \
            f"Round-trip failed: '{string_value}' -> {float_value} -> '{reconstructed}' -> {convertor.convert(reconstructed)} != {original_float}"
        print("Specific test passed")
    except AssertionError as e:
        print(f"Specific test failed: {e}")