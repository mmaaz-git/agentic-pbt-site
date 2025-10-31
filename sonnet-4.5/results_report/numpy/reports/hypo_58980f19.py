from hypothesis import given, settings, strategies as st
from numpy.polynomial import Polynomial
import numpy as np

polynomial_coefs = st.lists(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
    min_size=1,
    max_size=6
)

@given(polynomial_coefs)
@settings(max_examples=300)
def test_cast_to_same_type(coefs):
    p = Polynomial(coefs)
    p_cast = p.cast(Polynomial)

    assert np.allclose(p.coef, p_cast.coef, rtol=1e-10, atol=1e-10)

if __name__ == "__main__":
    test_cast_to_same_type()