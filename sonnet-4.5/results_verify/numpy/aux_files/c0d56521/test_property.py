from hypothesis import given, strategies as st, assume, settings
import numpy as np
from numpy.polynomial import Polynomial


small_poly_coefs = st.lists(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
    min_size=1,
    max_size=5
)


@given(small_poly_coefs, small_poly_coefs)
@settings(max_examples=1000)
def test_divmod_invariant(c1, c2):
    assume(len(c2) > 0)

    p2 = Polynomial(c2)
    p2_trimmed = p2.trim(tol=1e-10)
    assume(len(p2_trimmed.coef) > 0)
    assume(abs(p2_trimmed.coef[-1]) > 1e-100)

    p1 = Polynomial(c1)

    q, r = divmod(p1, p2)
    reconstructed = q * p2 + r

    p1_trimmed = p1.trim(tol=1e-10)
    reconstructed_trimmed = reconstructed.trim(tol=1e-10)

    assert np.allclose(reconstructed_trimmed.coef, p1_trimmed.coef, rtol=1e-7, atol=1e-7)

if __name__ == "__main__":
    test_divmod_invariant()
    print("Test passed!")