from hypothesis import given, strategies as st, settings, example
import numpy as np
import scipy.signal.windows as windows


@given(st.integers(min_value=1, max_value=100),
       st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@example(2, 2.225e-311)  # Add the specific failing case
@settings(max_examples=1000)
def test_tukey_no_nan(M, alpha):
    """Tukey window should never produce NaN for alpha in [0, 1]."""
    w = windows.tukey(M, alpha)
    assert not np.any(np.isnan(w)), \
        f"tukey({M}, alpha={alpha}) produced NaN: {w}"

if __name__ == "__main__":
    # Run the test
    print("Running Hypothesis test for scipy.signal.windows.tukey")
    print("=" * 60)
    try:
        test_tukey_no_nan()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")