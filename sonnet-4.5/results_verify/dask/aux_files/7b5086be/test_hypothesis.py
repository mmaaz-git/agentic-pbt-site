import math
from hypothesis import assume, given, strategies as st
from starlette.convertors import FloatConvertor

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e10))
def test_float_convertor_round_trip(x):
    convertor = FloatConvertor()
    assume(x >= 0.0)
    assume(not math.isnan(x))
    assume(not math.isinf(x))

    string_repr = convertor.to_string(x)
    result = convertor.convert(string_repr)

    assert math.isclose(result, x, rel_tol=1e-15), f"Round-trip failed: {x} -> {string_repr} -> {result}"

if __name__ == "__main__":
    test_float_convertor_round_trip()