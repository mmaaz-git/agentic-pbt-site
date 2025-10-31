from hypothesis import given, strategies as st
import math
from Cython.Utils import normalise_float_repr

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
def test_normalise_float_repr_preserves_value(f):
    float_str = str(f)
    result = normalise_float_repr(float_str)
    assert math.isclose(float(result), float(float_str), rel_tol=1e-14)

if __name__ == "__main__":
    test_normalise_float_repr_preserves_value()