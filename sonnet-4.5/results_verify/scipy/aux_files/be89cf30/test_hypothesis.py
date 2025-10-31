import numpy as np
from hypothesis import given, settings, strategies as st, example
from scipy.signal import windows

@given(M=st.integers(min_value=2, max_value=500),
       alpha=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
@example(M=2, alpha=5e-324)  # Add the specific failing case
def test_tukey_nonnegative(M, alpha):
    w = windows.tukey(M, alpha)
    assert np.all(~np.isnan(w)), f"Tukey window should not contain NaN values for M={M}, alpha={alpha}"
    assert np.all(w >= 0), f"Tukey window should be non-negative for M={M}, alpha={alpha}"

if __name__ == "__main__":
    test_tukey_nonnegative()