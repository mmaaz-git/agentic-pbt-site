import math
import numpy as np
from hypothesis import assume, given, settings, strategies as st
from scipy import integrate


@settings(max_examples=500)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=0.01, max_value=1e6), min_size=3, max_size=100),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
)
def test_simpson_constant(x_vals, c):
    x = np.sort(np.array(x_vals))
    y = np.full_like(x, c)

    result = integrate.simpson(y, x=x)
    expected = c * (x[-1] - x[0])

    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-9)

if __name__ == "__main__":
    # Test with the specific failing input
    x_vals = [1.0, 1.0, 2.0]
    c = 1.0

    x = np.sort(np.array(x_vals))
    y = np.full_like(x, c)

    result = integrate.simpson(y, x=x)
    expected = c * (x[-1] - x[0])

    print(f"Testing with x_vals={x_vals}, c={c}")
    print(f"x array: {x}")
    print(f"y array: {y}")
    print(f"simpson result: {result}")
    print(f"Expected: {expected}")
    print(f"Test passes: {math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-9)}")