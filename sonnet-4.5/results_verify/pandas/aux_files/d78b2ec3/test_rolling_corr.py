import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume, settings

# Property-based test from the bug report
@settings(max_examples=200)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3), min_size=5, max_size=50),
    st.integers(min_value=2, max_value=10)
)
def test_rolling_corr_bounds(data, window):
    assume(len(data) >= window)
    assume(len(set(data)) > 1)

    s = pd.Series(data)
    corr = s.rolling(window=window).corr()

    valid = corr[~corr.isna()]
    if len(valid) > 0:
        assert (valid >= -1.0001).all(), f"Found correlation < -1: {valid.min()}"
        assert (valid <= 1.0001).all(), f"Found correlation > 1: {valid.max()}"

# Test with the specific failing input
print("Testing with the specific failing input from bug report:")
data = [0.0, 0.0, 0.0, 0.0, 7.797011399495068e-124]
window = 2
s = pd.Series(data)
corr = s.rolling(window=window).corr()
print(f"Data: {data}")
print(f"Window: {window}")
print(f"Rolling correlation result: {corr}")
print(f"Contains inf: {np.isinf(corr).any()}")
print()

# Run the hypothesis test
print("Running property-based test...")
try:
    test_rolling_corr_bounds()
    print("Property-based test passed!")
except AssertionError as e:
    print(f"Property-based test failed: {e}")