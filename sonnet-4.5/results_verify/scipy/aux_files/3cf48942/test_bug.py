#!/usr/bin/env python3
from hypothesis import given, strategies as st, assume, settings
import numpy as np
from scipy.integrate import cumulative_simpson

# Property-based test from the bug report
@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=3, max_size=100)
)
@settings(max_examples=500)
def test_cumulative_simpson_monotonic_increasing_for_positive(y):
    y_arr = np.array(y)
    assume(np.all(y_arr >= 0))
    assume(np.any(y_arr > 0))

    cumulative_result = cumulative_simpson(y_arr, initial=0)

    diffs = np.diff(cumulative_result)
    assert np.all(diffs >= -1e-10)

# Run the specific failing case
print("=" * 50)
print("Testing specific failing case from bug report:")
print("=" * 50)

y = np.array([0.0, 0.0, 1.0])

cumulative_result = cumulative_simpson(y, initial=0)
diffs = np.diff(cumulative_result)

print(f"y = {y}")
print(f"cumulative_simpson(y, initial=0) = {cumulative_result}")
print(f"Differences between consecutive values: {diffs}")
print(f"Has negative difference: {np.any(diffs < 0)}")
print()

# Try running the hypothesis test to find failures
print("=" * 50)
print("Running hypothesis test to find failures:")
print("=" * 50)
try:
    test_cumulative_simpson_monotonic_increasing_for_positive()
    print("No failures found!")
except AssertionError as e:
    print(f"Found failure in hypothesis test")