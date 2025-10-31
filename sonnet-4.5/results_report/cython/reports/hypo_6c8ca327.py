from hypothesis import given, strategies as st, settings
from Cython.Utils import normalise_float_repr

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
@settings(max_examples=1000)
def test_normalise_float_repr_preserves_value(f):
    float_str = str(f)
    result = normalise_float_repr(float_str)

    original_value = float(float_str)
    result_value = float(result)

    assert original_value == result_value

# Run the test
if __name__ == "__main__":
    test_normalise_float_repr_preserves_value()