import numpy as np
import pandas as pd
from xarray.plot.utils import _interval_to_bound_points

# Test case 1: Non-contiguous intervals from the bug report
print("Test 1: Non-contiguous intervals")
intervals = [
    pd.Interval(0, 2),
    pd.Interval(5, 7),
    pd.Interval(10, 12)
]

result = _interval_to_bound_points(intervals)
print(f"Input intervals: {intervals}")
print(f"Result: {result}")
print(f"Expected (according to bug report): [0, 2, 5, 7, 10, 12] (all boundaries)")
print(f"Actual result shape: {result.shape}")
print()

# Test case 2: Contiguous intervals
print("Test 2: Contiguous intervals")
contiguous_intervals = [
    pd.Interval(0, 2),
    pd.Interval(2, 5),
    pd.Interval(5, 7)
]

result2 = _interval_to_bound_points(contiguous_intervals)
print(f"Input intervals: {contiguous_intervals}")
print(f"Result: {result2}")
print(f"Expected: [0, 2, 5, 7] (boundaries for contiguous intervals)")
print()

# Test case 3: Property-based test
print("Test 3: Property-based test from bug report")
from hypothesis import given, strategies as st

@st.composite
def non_contiguous_intervals(draw):
    n = draw(st.integers(min_value=2, max_value=10))
    intervals = []
    current_pos = 0
    for i in range(n):
        left = current_pos
        width = draw(st.floats(min_value=0.1, max_value=10))
        right = left + width
        intervals.append(pd.Interval(left, right))
        gap = draw(st.floats(min_value=0.1, max_value=5))
        current_pos = right + gap
    return intervals

@given(non_contiguous_intervals())
def test_interval_to_bound_points_completeness(intervals):
    result = _interval_to_bound_points(intervals)

    expected_bounds = []
    for interval in intervals:
        expected_bounds.append(interval.left)
    expected_bounds.append(intervals[-1].right)

    # Check if intervals are contiguous
    if not all(intervals[i].right == intervals[i+1].left for i in range(len(intervals)-1)):
        actual_all_bounds = []
        for interval in intervals:
            actual_all_bounds.extend([interval.left, interval.right])

        for bound in actual_all_bounds:
            assert bound in result, f"Missing boundary {bound} in result {result}"

# Run a simple test without hypothesis
test_intervals = non_contiguous_intervals().example()
print(f"Generated test intervals: {test_intervals}")
result3 = _interval_to_bound_points(test_intervals)
print(f"Result: {result3}")

# Check if the result contains all boundaries
actual_all_bounds = []
for interval in test_intervals:
    actual_all_bounds.extend([interval.left, interval.right])
actual_all_bounds = sorted(set(actual_all_bounds))
print(f"All actual boundaries: {actual_all_bounds}")
print(f"Is result missing boundaries? {set(actual_all_bounds) - set(result3)}")