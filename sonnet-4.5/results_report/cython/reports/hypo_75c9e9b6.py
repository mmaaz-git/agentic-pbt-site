from hypothesis import given, strategies as st, settings
from Cython.Utils import normalise_float_repr


@given(st.floats(allow_nan=False, allow_infinity=False,
                 min_value=-1e50, max_value=1e50))
@settings(max_examples=1000)
def test_round_trip_property(f):
    """
    Property: normalise_float_repr should preserve the numerical value.
    For any valid float string, the normalized form should represent the same number.
    """
    float_str = str(f)
    result = normalise_float_repr(float_str)

    try:
        parsed_result = float(result)
    except ValueError as e:
        print(f"\nValueError when parsing result: {e}")
        print(f"Input: {float_str}")
        print(f"Result: {result}")
        raise AssertionError(f"Result '{result}' is not a valid float string")

    assert parsed_result == float(float_str), \
        f"Value changed: {float_str} -> {result} ({float(float_str)} != {parsed_result})"

if __name__ == "__main__":
    test_round_trip_property()