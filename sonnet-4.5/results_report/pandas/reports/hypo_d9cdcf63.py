import numpy as np
from hypothesis import given, strategies as st, settings
import scipy.signal.windows as w

@given(
    M=st.integers(min_value=2, max_value=500),
    alpha=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=300)
def test_tukey_symmetry(M, alpha):
    window = w.tukey(M, alpha, sym=True)
    assert np.allclose(window, window[::-1]), \
        f"tukey({M}, {alpha}, sym=True) is not symmetric"

if __name__ == "__main__":
    test_tukey_symmetry()