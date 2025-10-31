import pandas as pd
from pandas.core.arrays import IntervalArray

# Create an IntervalArray with negative breaks
arr = IntervalArray.from_breaks([-3, -2, -1], closed='left')
print(f"Original intervals: {list(arr)}")

# Concatenate the array with itself (to simulate having duplicates)
combined = IntervalArray._concat_same_type([arr, arr])
print(f"Concatenated: {list(combined)}")

# Apply unique() to get distinct intervals
unique = combined.unique()
print(f"After unique(): {list(unique)}")
print(f"Length: {len(unique)} (expected 2)")

# Verify the bug: we expect 2 unique intervals but get only 1
expected_intervals = 2
actual_intervals = len(unique)
print(f"\nBUG: Expected {expected_intervals} unique intervals, got {actual_intervals}")