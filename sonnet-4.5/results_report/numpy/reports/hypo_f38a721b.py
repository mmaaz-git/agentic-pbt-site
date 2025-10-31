import numpy as np
from hypothesis import given, settings, strategies as st
from numpy.polynomial import Polynomial


@settings(max_examples=500)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=1, max_size=6),
    st.integers(min_value=2, max_value=5)
)
def test_power_equals_repeated_multiplication(coeffs, n):
    p = Polynomial(coeffs)

    power_result = p ** n

    mult_result = Polynomial([1])
    for _ in range(n):
        mult_result = mult_result * p

    # Check if coefficient arrays have the same shape first
    if power_result.coef.shape != mult_result.coef.shape:
        raise AssertionError(
            f"Coefficient shapes differ: p**{n} has shape {power_result.coef.shape}, "
            f"p*...*p has shape {mult_result.coef.shape}. "
            f"Coeffs: {coeffs}, n: {n}"
        )

    assert np.allclose(power_result.coef, mult_result.coef, atol=1e-8), \
        f"p**{n} != p*...*p (repeated {n} times)"

# Run the test
test_power_equals_repeated_multiplication()