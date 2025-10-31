import numpy as np
from hypothesis import given, strategies as st, settings
import scipy.signal.windows as windows


@given(st.integers(min_value=3, max_value=100).filter(lambda x: x % 2 == 1))
@settings(max_examples=50)
def test_flattop_normalization_bug(M):
    """
    The flattop window claims to return a window "with the maximum value
    normalized to 1", but for odd M values, the maximum value exceeds 1.0.
    """
    w = windows.flattop(M)
    max_val = np.max(w)

    assert max_val <= 1.0, \
        f"flattop({M}) has max {max_val:.15f} > 1.0 (exceeds by {max_val - 1.0:.3e})"

if __name__ == "__main__":
    test_flattop_normalization_bug()