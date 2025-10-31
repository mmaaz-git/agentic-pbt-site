import math
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy.integrate import tanhsinh

@given(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100)
)
@settings(max_examples=300)
def test_tanhsinh_constant(c, a, b):
    assume(abs(b - a) > 0.01)
    assume(abs(a) < 100 and abs(b) < 100)

    def f(x):
        return c

    result = tanhsinh(f, a, b)
    expected = c * (b - a)

    assert math.isclose(result.integral, expected, rel_tol=1e-8, abs_tol=1e-10)

if __name__ == "__main__":
    test_tanhsinh_constant()