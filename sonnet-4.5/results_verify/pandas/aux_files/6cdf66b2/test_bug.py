import numpy as np
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

print("=== Testing FixedForwardWindowIndexer with negative window_size ===")
print()

# Test 1: Basic reproduction from the bug report
print("Test 1: Basic case with window_size=-1")
indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2, step=1)

print(f"start: {start}")
print(f"end: {end}")
print(f"Invariant check (start <= end): {np.all(start <= end)}")
print()

# Test 2: With larger negative window
print("Test 2: Larger negative window_size=-5")
indexer2 = FixedForwardWindowIndexer(window_size=-5)
start2, end2 = indexer2.get_window_bounds(num_values=10, step=1)
print(f"start: {start2}")
print(f"end: {end2}")
print(f"Invariant check (start <= end): {np.all(start2 <= end2)}")
print()

# Test 3: Using with actual rolling operations
print("Test 3: Using with DataFrame.rolling()")
df = pd.DataFrame({'values': range(10)})
indexer3 = FixedForwardWindowIndexer(window_size=-5)
result = df.rolling(indexer3).sum()
print("DataFrame:")
print(df)
print("\nResult with window_size=-5:")
print(result)
print()

# Test 4: Property-based test from the report
print("Test 4: Running the hypothesis-like test")
from hypothesis import given, strategies as st

@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-100, max_value=100),
    step=st.integers(min_value=1, max_value=10),
)
def test_fixed_forward_start_le_end_always(num_values, window_size, step):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values, step=step)
    assert np.all(start <= end), f"Invariant violated: start > end for window_size={window_size}"

# Run the test with specific failing input
try:
    test_fixed_forward_start_le_end_always(num_values=2, window_size=-1, step=1)
    print("Hypothesis test passed")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")