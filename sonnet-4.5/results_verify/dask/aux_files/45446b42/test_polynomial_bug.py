import numpy as np
import numpy.polynomial as poly
from hypothesis import given, settings, strategies as st


@settings(max_examples=500)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3), min_size=1, max_size=5),
    st.integers(min_value=2, max_value=3)
)
def test_power_by_repeated_multiplication(coeffs, n):
    p = poly.Polynomial(coeffs)

    result_power = p ** n
    result_mult = p
    for _ in range(n - 1):
        result_mult = result_mult * p

    np.testing.assert_array_equal(result_power.coef, result_mult.coef)

# Run the test with the specific failing input
print("Testing with specific failing input:")
coeffs = [0.0, 1.1125369292536007e-308]
n = 2

p = poly.Polynomial(coeffs)
result_power = p ** n
result_mult = p * p

print(f"Input coefficients: {coeffs}")
print(f"Power: {n}")
print(f"p**{n} coefficients: {result_power.coef}")
print(f"p*p coefficients: {result_mult.coef}")
print(f"Arrays equal: {np.array_equal(result_power.coef, result_mult.coef)}")
print()

# Try running the hypothesis test
print("Running hypothesis test...")
try:
    test_power_by_repeated_multiplication()
    print("Hypothesis test passed!")
except AssertionError as e:
    print(f"Hypothesis test failed with error: {e}")
except Exception as e:
    print(f"Hypothesis test failed with unexpected error: {e}")