import scipy.special as sp
import numpy as np
from hypothesis import given, strategies as st, settings

@settings(max_examples=2000)
@given(st.floats(allow_nan=False, allow_infinity=False, min_value=1e-308, max_value=10),
       st.floats(allow_nan=False, allow_infinity=False, min_value=-10, max_value=10))
def test_pseudo_huber_returns_finite(delta, r):
    result = sp.pseudo_huber(delta, r)
    assert np.isfinite(result), f"pseudo_huber({delta}, {r}) returned {result}, expected finite value"

if __name__ == "__main__":
    test_pseudo_huber_returns_finite()