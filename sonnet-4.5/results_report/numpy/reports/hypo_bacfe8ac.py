from hypothesis import assume, given, settings, strategies as st
import numpy as np
from numpy.polynomial import Polynomial

@st.composite
def polynomial_coefficients(draw):
    size = draw(st.integers(min_value=1, max_value=10))
    coefs = draw(st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=size,
        max_size=size
    ))
    coefs = [c if abs(c) >= 1e-10 else 0.0 for c in coefs]
    assume(any(c != 0 for c in coefs))
    return coefs

@given(polynomial_coefficients(), polynomial_coefficients())
@settings(max_examples=500)
def test_polynomial_divmod_property(coefs_a, coefs_b):
    a = Polynomial(coefs_a)
    b = Polynomial(coefs_b)
    assume(not np.allclose(b.coef, 0))

    q, r = divmod(a, b)
    reconstructed = b * q + r

    assert np.allclose(reconstructed.trim().coef, a.trim().coef, rtol=1e-5, atol=1e-5)

def test_specific_case(coefs_a, coefs_b):
    """Test a specific case"""
    a = Polynomial(coefs_a)
    b = Polynomial(coefs_b)

    q, r = divmod(a, b)
    reconstructed = b * q + r

    if not np.allclose(reconstructed.trim().coef, a.trim().coef, rtol=1e-5, atol=1e-5):
        raise AssertionError(f"Test failed: max error = {np.max(np.abs(reconstructed.trim().coef - a.coef))}")

if __name__ == "__main__":
    # Run the test with specific failing example
    print("Testing with failing example from initial report:")
    print("coefs_a = [0, 0, 0, 0, 0, 0, 0, 1]")
    print("coefs_b = [72, 1.75]")
    print()

    try:
        test_specific_case([0, 0, 0, 0, 0, 0, 0, 1], [72, 1.75])
        print("Test passed for initial example")
    except AssertionError as e:
        print("Test FAILED for initial example")
        print(f"  {e}")

    print("\n" + "="*60)
    print("Running property-based test with Hypothesis...")
    print("="*60 + "\n")

    # Run the full hypothesis test
    from hypothesis import reproduce_failure, __version__ as hypothesis_version
    import traceback

    try:
        test_polynomial_divmod_property()
        print("All tests passed!")
    except Exception as e:
        print("Test failed with Hypothesis-found example:")
        # Print last part of the traceback
        traceback.print_exc()