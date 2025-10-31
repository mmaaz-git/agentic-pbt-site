#!/usr/bin/env python3
import numpy as np
import numpy.polynomial as np_poly
from hypothesis import assume, given, settings, strategies as st


@settings(max_examples=1000)
@given(
    coef=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
        min_size=2,
        max_size=6
    )
)
def test_polynomial_roots_are_valid(coef):
    p = np_poly.Polynomial(coef)

    assume(p.degree() >= 1)

    roots = p.roots()

    assume(not np.any(np.isnan(roots)))
    assume(not np.any(np.isinf(roots)))

    for root in roots:
        value = abs(p(root))
        assert value < 1e-6, f'p({root}) = {p(root)}, expected ~0'


if __name__ == "__main__":
    # Run the test
    test_polynomial_roots_are_valid()