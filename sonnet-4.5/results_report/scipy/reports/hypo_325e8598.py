import numpy as np
from hypothesis import given, strategies as st, settings
import scipy.signal.windows as w

@given(st.integers(min_value=2, max_value=1000))
@settings(max_examples=500)
def test_blackman_non_negative(M):
    """Window values should be non-negative."""
    result = w.blackman(M)
    assert np.all(result >= 0), \
        f"blackman(M={M}) has negative values: min={np.min(result)}"

if __name__ == "__main__":
    # Run the test
    test_blackman_non_negative()