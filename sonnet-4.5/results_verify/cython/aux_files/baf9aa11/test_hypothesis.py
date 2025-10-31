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
        normalized_value = float(result)
        original_value = float(float_str)
        assert normalized_value == original_value, \
            f"Value changed: {float_str} -> {result} ({original_value} != {normalized_value})"
    except ValueError as e:
        raise AssertionError(f"Result '{result}' is not a valid float! Original: {float_str}")

# Run the test
test_round_trip_property()
print("Test completed without failures")