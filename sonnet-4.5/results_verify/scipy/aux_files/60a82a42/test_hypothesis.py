from hypothesis import given, strategies as st
import scipy.special as sp
import math

@given(
    st.floats(min_value=1e-10, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
)
def test_boxcox1p_inv_boxcox1p_roundtrip(x, lmbda):
    """Test that inv_boxcox1p(boxcox1p(x, lmbda), lmbda) = x"""
    y = sp.boxcox1p(x, lmbda)
    result = sp.inv_boxcox1p(y, lmbda)
    assert math.isclose(result, x, rel_tol=1e-7, abs_tol=1e-10), \
        f"inv_boxcox1p(boxcox1p({x}, {lmbda}), {lmbda}) = {result}, expected {x}"

# Also test the specific failing case mentioned
def test_specific_case():
    x = 1.0
    lmbda = 3.9731703875764937e-287
    y = sp.boxcox1p(x, lmbda)
    result = sp.inv_boxcox1p(y, lmbda)
    print(f"Testing specific case: x={x}, lmbda={lmbda}")
    print(f"boxcox1p({x}, {lmbda}) = {y}")
    print(f"inv_boxcox1p({y}, {lmbda}) = {result}")
    print(f"Expected: {x}")
    print(f"Actual: {result}")
    print(f"Error: {abs(result - x)}")
    assert math.isclose(result, x, rel_tol=1e-7, abs_tol=1e-10), \
        f"inv_boxcox1p(boxcox1p({x}, {lmbda}), {lmbda}) = {result}, expected {x}"

if __name__ == "__main__":
    # Run specific test case first
    try:
        test_specific_case()
        print("\nSpecific test case passed!")
    except AssertionError as e:
        print(f"\nSpecific test case failed: {e}")

    # Run hypothesis tests
    print("\nRunning Hypothesis tests...")
    test_boxcox1p_inv_boxcox1p_roundtrip()
    print("Hypothesis tests completed!")