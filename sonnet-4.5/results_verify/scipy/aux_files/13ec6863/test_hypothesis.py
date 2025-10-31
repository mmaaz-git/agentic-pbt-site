from hypothesis import given, strategies as st, example
import numpy as np
import scipy.signal.windows as windows


@given(st.integers(min_value=1, max_value=100),
       st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@example(2, 2.225e-311)  # The reported failing input
def test_tukey_no_nan(M, alpha):
    """Tukey window should never produce NaN for alpha in [0, 1]."""
    w = windows.tukey(M, alpha)
    print(f"Testing: M={M}, alpha={alpha:.2e}, result={w}")
    assert not np.any(np.isnan(w)), \
        f"tukey({M}, alpha={alpha}) produced NaN: {w}"

if __name__ == "__main__":
    # Run a simple direct test with the reported failing input
    M = 2
    alpha = 2.225e-311
    w = windows.tukey(M, alpha)
    print(f"Direct test: M={M}, alpha={alpha:.2e}, result={w}")
    if np.any(np.isnan(w)):
        print(f"Test failed: tukey({M}, alpha={alpha}) produced NaN: {w}")
    else:
        print("Test passed with reported failing input")