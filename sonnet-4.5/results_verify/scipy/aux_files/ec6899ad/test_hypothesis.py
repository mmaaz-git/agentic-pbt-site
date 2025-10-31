import numpy as np
from hypothesis import given, strategies as st, example
from scipy.signal.windows import tukey


@given(st.integers(min_value=2, max_value=100),
       st.floats(min_value=1e-320, max_value=1e-100, allow_nan=False, allow_infinity=False))
@example(2, 1e-320)  # The specific failing case mentioned
def test_tukey_no_nan_with_tiny_alpha(M, alpha):
    w = tukey(M, alpha=alpha, sym=True)

    assert len(w) == M, f"Expected length {M}, got {len(w)}"
    assert np.all(np.isfinite(w)), \
        f"tukey({M}, alpha={alpha}) contains non-finite values: {w}"

# Run the test
if __name__ == "__main__":
    test_tukey_no_nan_with_tiny_alpha()
    print("Test completed")