import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=5, max_size=30),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=5, max_size=30)
)
def test_rolling_corr_bounds(values1, values2):
    assume(len(values1) == len(values2))

    s1 = pd.Series(values1)
    s2 = pd.Series(values2)

    result = s1.rolling(3).corr(s2)

    for i, val in enumerate(result):
        if not np.isnan(val):
            assert -1 <= val <= 1, f"At index {i}: correlation {val} outside [-1, 1]"

# Test with the specific failing input
print("Testing with the failing input...")
values1 = [0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0]
values2 = values1

try:
    test_rolling_corr_bounds(values1, values2)
    print("Test passed")
except AssertionError as e:
    print(f"Test failed: {e}")

# Run the hypothesis test
print("\nRunning hypothesis test (limited runs)...")
import hypothesis
hypothesis.settings(max_examples=100)

try:
    test_rolling_corr_bounds()
    print("Hypothesis test passed")
except Exception as e:
    print(f"Hypothesis test failed: {e}")