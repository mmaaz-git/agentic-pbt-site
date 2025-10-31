import math
from hypothesis import given, strategies as st, settings
from scipy import special

@given(
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-2, max_value=2, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=1000)
def test_boxcox_inv_boxcox_round_trip(x, lmbda):
    y = special.boxcox(x, lmbda)
    result = special.inv_boxcox(y, lmbda)
    assert math.isclose(result, x, rel_tol=1e-9, abs_tol=1e-9)

# Test with the specific failing input
print("Testing with specific failing input: x=0.5, lmbda=5e-324")
try:
    test_boxcox_inv_boxcox_round_trip(0.5, 5e-324)
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")

    # Show the actual values
    x = 0.5
    lmbda = 5e-324
    y = special.boxcox(x, lmbda)
    result = special.inv_boxcox(y, lmbda)
    print(f"boxcox({x}, {lmbda}) = {y}")
    print(f"inv_boxcox({y}, {lmbda}) = {result}")
    print(f"Expected result to be {x}, but got {result}")
    print(f"Difference: {abs(result - x)}")

# Run the full hypothesis test
print("\nRunning full Hypothesis test...")
try:
    test_boxcox_inv_boxcox_round_trip()
    print("All tests passed!")
except Exception as e:
    print(f"Test failed: {e}")