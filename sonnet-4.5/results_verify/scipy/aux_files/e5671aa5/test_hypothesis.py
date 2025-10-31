from hypothesis import given, strategies as st, settings
import scipy.signal.windows as windows
import numpy as np

@settings(max_examples=300)
@given(
    M=st.integers(min_value=3, max_value=1000),
    alpha=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)
)
def test_tukey_no_nan(M, alpha):
    result = windows.tukey(M, alpha=alpha, sym=True)
    assert not np.any(np.isnan(result)), f"tukey({M}, alpha={alpha}) contains NaN"
    assert not np.any(np.isinf(result)), f"tukey({M}, alpha={alpha}) contains inf"

if __name__ == "__main__":
    test_tukey_no_nan()
    print("Test passed!")