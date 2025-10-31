import numpy as np
from scipy import integrate
from hypothesis import given, strategies as st, settings

@given(
    n=st.sampled_from([3, 5, 7, 9, 11]),
    a=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=200)
def test_simpson_reversal_odd_n(n, a, b):
    from hypothesis import assume
    assume(b > a)
    assume(abs(b - a) > 1e-6)

    x_forward = np.linspace(a, b, n)
    y = np.random.randn(n)

    x_backward = x_forward[::-1]
    y_backward = y[::-1]

    result_forward = integrate.simpson(y, x=x_forward)
    result_backward = integrate.simpson(y_backward, x=x_backward)

    assert np.isclose(result_forward, -result_backward, rtol=1e-10, atol=1e-10), f"Forward: {result_forward}, Backward: {result_backward}, n={n}"

# Run the test
if __name__ == "__main__":
    test_simpson_reversal_odd_n()
    print("Test passed for odd n values")