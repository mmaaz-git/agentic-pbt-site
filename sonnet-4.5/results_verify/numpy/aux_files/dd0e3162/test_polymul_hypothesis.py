import numpy as np
from hypothesis import given, settings, strategies as st
from numpy.polynomial import polynomial as P

coefficients = st.lists(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    min_size=1,
    max_size=10
).map(lambda x: np.array(x))

@settings(max_examples=500)
@given(c1=coefficients, c2=coefficients, c3=coefficients)
def test_polymul_associative(c1, c2, c3):
    result1 = P.polymul(P.polymul(c1, c2), c3)
    result2 = P.polymul(c1, P.polymul(c2, c3))
    np.testing.assert_allclose(result1, result2, rtol=1e-9, atol=1e-9)

if __name__ == "__main__":
    test_polymul_associative()