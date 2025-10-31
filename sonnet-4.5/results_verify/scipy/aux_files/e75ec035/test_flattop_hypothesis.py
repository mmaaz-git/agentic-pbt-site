import numpy as np
from hypothesis import given, strategies as st, settings
import scipy.signal.windows as windows


@given(st.integers(min_value=1, max_value=1000))
@settings(max_examples=300)
def test_normalization_property(M):
    w = windows.flattop(M)
    max_val = np.max(w)

    assert max_val <= 1.0, f"flattop({M}) has max value {max_val} > 1.0"

# Run the test
print("Running Hypothesis test...")
try:
    test_normalization_property()
    print("All tests passed!")
except AssertionError as e:
    print(f"Assertion failed: {e}")
except Exception as e:
    print(f"Test failed: {e}")