from hypothesis import given, strategies as st
import scipy.optimize.cython_optimize._zeros as zeros


@given(
    c0=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    c1=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    c2=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    c3=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False).filter(lambda x: abs(x) > 0.01),
)
def test_polynomial_form_documented(c0, c1, c2, c3):
    root = zeros.EXAMPLES_MAP['brentq']((c0, c1, c2, c3), -10.0, 10.0, 1e-6, 1e-6, 100)

    poly_val = c0 + c1*root + c2*root**2 + c3*root**3
    assert abs(poly_val) < 1e-3, f"Polynomial value at root should be near zero, got {poly_val}"

if __name__ == "__main__":
    # Run the test
    test_polynomial_form_documented()
    print("Test passed!")