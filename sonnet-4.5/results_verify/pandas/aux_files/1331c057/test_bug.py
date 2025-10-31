import numpy as np
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer

# First test the specific failing case
print("=== Testing specific failing case ===")
indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2)

print(f"start: {start}")
print(f"end: {end}")
print(f"end >= start: {end >= start}")
print(f"All end >= start: {np.all(end >= start)}")

# Verify the assertion in the bug report
assert not np.all(end >= start), "Bug report assertion confirmed"
print("Bug confirmed: end < start for some indices\n")

# Test with other negative values
print("=== Testing other negative window sizes ===")
for window_size in [-5, -2, -1]:
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=5)
    print(f"window_size={window_size}: start={start}, end={end}")
    print(f"  All end >= start: {np.all(end >= start)}")

# Test with positive values for comparison
print("\n=== Testing positive window sizes (expected behavior) ===")
for window_size in [0, 1, 2, 5]:
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=5)
    print(f"window_size={window_size}: start={start}, end={end}")
    print(f"  All end >= start: {np.all(end >= start)}")

# Run the hypothesis test
print("\n=== Running property-based test ===")
@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-100, max_value=100)
)
def test_fixed_forward_window_negative_size(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)

    assert len(start) == len(end)
    # This assertion will fail for negative window_size
    if window_size < 0:
        # We expect this to fail
        return not np.all(end >= start)
    else:
        assert np.all(end >= start)
    return True

# Try to run hypothesis test
try:
    test_fixed_forward_window_negative_size()
    print("Hypothesis test passed (modified to expect failure for negative sizes)")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

# Test practical impact with rolling operations
print("\n=== Testing practical impact with rolling operations ===")
import pandas as pd

df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})

# Test with positive window (expected behavior)
indexer_pos = FixedForwardWindowIndexer(window_size=2)
result_pos = df.rolling(window=indexer_pos, min_periods=1).sum()
print("Positive window_size=2:")
print(result_pos)

# Test with negative window (problematic behavior)
try:
    indexer_neg = FixedForwardWindowIndexer(window_size=-1)
    result_neg = df.rolling(window=indexer_neg, min_periods=1).sum()
    print("\nNegative window_size=-1:")
    print(result_neg)
except Exception as e:
    print(f"Error with negative window: {e}")