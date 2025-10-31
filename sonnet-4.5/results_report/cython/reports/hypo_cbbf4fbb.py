from hypothesis import given, strategies as st, settings
from Cython.Utils import normalise_float_repr
import math

@settings(max_examples=1000)
@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
def test_normalise_float_repr_value_preservation(x):
    float_str = str(x)
    normalized = normalise_float_repr(float_str)
    assert math.isclose(float(normalized), float(float_str), rel_tol=1e-15)

if __name__ == "__main__":
    test_normalise_float_repr_value_preservation()