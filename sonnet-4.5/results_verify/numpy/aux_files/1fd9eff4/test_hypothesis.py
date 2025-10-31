import numpy as np
from hypothesis import given, settings, strategies as st, example
from numpy.polynomial import Polynomial


@st.composite
def polynomials_safe(draw, max_degree=8):
    deg = draw(st.integers(min_value=0, max_value=max_degree))
    coef = draw(st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=deg+1, max_size=deg+1
    ))
    return Polynomial(coef)


@given(polynomials_safe(), polynomials_safe())
@settings(max_examples=500)
@example(Polynomial([0., 1.]), Polynomial([1.0, 2.22507386e-311]))
def test_divmod_no_overflow(a, b):
    if np.allclose(b.coef, 0):
        return

    try:
        q, r = divmod(a, b)
        assert not np.any(np.isnan(q.coef)) and not np.any(np.isnan(r.coef))
        assert not np.any(np.isinf(q.coef)) and not np.any(np.isinf(r.coef))
    except ZeroDivisionError:
        pass

if __name__ == "__main__":
    test_divmod_no_overflow()