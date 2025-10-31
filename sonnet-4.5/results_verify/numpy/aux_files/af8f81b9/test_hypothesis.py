import numpy as np
from numpy.polynomial import polynomial as poly
from hypothesis import given, strategies as st, settings, example
from numpy.polynomial import polyutils

@given(
    st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=1, max_size=3),
    st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=1, max_size=3)
)
@example([1.0, 1.0], [1.0, 1e-100])  # The specific failing case
@settings(max_examples=100)
def test_polydiv_reconstruction_property(dividend, divisor):
    from hypothesis import assume

    divisor_trimmed = polyutils.trimcoef(divisor)
    assume(len(divisor_trimmed) > 0)
    assume(not np.allclose(divisor_trimmed, 0))

    quo, rem = poly.polydiv(dividend, divisor)
    reconstructed = poly.polyadd(poly.polymul(quo, divisor), rem)

    dividend_trimmed = polyutils.trimcoef(dividend)
    reconstructed_trimmed = polyutils.trimcoef(reconstructed)

    print(f"\nTesting: dividend={dividend}, divisor={divisor}")
    print(f"  Quo: {quo}, Rem: {rem}")
    print(f"  Reconstructed: {reconstructed}")

    assert len(dividend_trimmed) == len(reconstructed_trimmed), \
        f"Length mismatch: {len(dividend_trimmed)} != {len(reconstructed_trimmed)}"

    for i, (orig, recon) in enumerate(zip(dividend_trimmed, reconstructed_trimmed)):
        if not np.isclose(orig, recon, rtol=1e-6, atol=1e-9):
            print(f"  Failed at index {i}: {orig} != {recon}")
            print(f"  Difference: {abs(orig - recon)}")
            raise AssertionError(f"Values not close at index {i}: {orig} != {recon}")

if __name__ == "__main__":
    # Test the specific failing case
    print("Testing the specific case from the bug report:")
    try:
        test_polydiv_reconstruction_property([1.0, 1.0], [1.0, 1e-100])
        print("✓ Test passed")
    except AssertionError as e:
        print(f"✗ Test failed: {e}")

    print("\n" + "=" * 60)
    print("Running hypothesis tests...")
    try:
        test_polydiv_reconstruction_property()
        print("✓ All tests passed")
    except Exception as e:
        print(f"✗ Tests failed: {e}")