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
for i in range(len(start)):
    print(f"  Window {i}: start={start[i]}, end={end[i]}, valid={start[i] <= end[i]}")
print()

# Test 2: With larger negative window
print("Test 2: Larger negative window_size=-5")
indexer2 = FixedForwardWindowIndexer(window_size=-5)
start2, end2 = indexer2.get_window_bounds(num_values=10, step=1)
print(f"start: {start2}")
print(f"end: {end2}")
print(f"Invariant check (start <= end): {np.all(start2 <= end2)}")
violations = []
for i in range(len(start2)):
    if start2[i] > end2[i]:
        violations.append(f"Window {i}: start={start2[i]} > end={end2[i]}")
if violations:
    print("Violations found:")
    for v in violations[:5]:  # Show first 5 violations
        print(f"  {v}")
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
print("Note: All values are 0.0, which is incorrect for a forward window sum")
print()

# Test 4: Test with positive window size for comparison
print("Test 4: Comparison with positive window_size=2")
indexer4 = FixedForwardWindowIndexer(window_size=2)
start4, end4 = indexer4.get_window_bounds(num_values=5, step=1)
print(f"start: {start4}")
print(f"end: {end4}")
print(f"Invariant check (start <= end): {np.all(start4 <= end4)}")
df2 = pd.DataFrame({'values': range(5)})
result2 = df2.rolling(indexer4).sum()
print("\nDataFrame:")
print(df2)
print("\nResult with window_size=2:")
print(result2)
print()

# Test 5: Zero window size
print("Test 5: window_size=0")
indexer5 = FixedForwardWindowIndexer(window_size=0)
start5, end5 = indexer5.get_window_bounds(num_values=3, step=1)
print(f"start: {start5}")
print(f"end: {end5}")
print(f"Invariant check (start <= end): {np.all(start5 <= end5)}")
print()

# Test 6: Check the logic in get_window_bounds
print("Test 6: Understanding the logic")
print("For negative window_size, let's trace through the logic:")
print("  start = np.arange(0, num_values, step)")
print("  end = start + self.window_size")
print("  if self.window_size:")
print("      end = np.clip(end, 0, num_values)")
print()
print("With window_size=-5 and num_values=10:")
print("  start = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]")
print("  end (before clip) = start + (-5) = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]")
print("  end (after clip to [0, 10]) = [0, 0, 0, 0, 0, 0, 1, 2, 3, 4]")
print()
print("This creates invalid windows where start > end for indices 1-5")