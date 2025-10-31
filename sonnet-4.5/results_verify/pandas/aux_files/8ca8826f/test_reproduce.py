import pandas as pd
import numpy as np

print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print()

# Reproduce the bug as described
series = pd.Series([
    1.000000e+00,
    1.605551e-178,
    -2.798597e-225,
    -2.225074e-308,
    -2.798597e-225,
])

rolling_mean = series.rolling(window=2).mean()

print("Series values:")
for i, val in enumerate(series):
    print(f"  Index {i}: {val}")
print()

print("Rolling mean (window=2):")
for i, val in enumerate(rolling_mean):
    print(f"  Index {i}: {val}")
print()

# Focus on the problematic window at index 3
print("Detailed analysis at index 3:")
print(f"Window at index 3: {series.iloc[2:4].values}")
print(f"  Value 1: {series.iloc[2]}")
print(f"  Value 2: {series.iloc[3]}")
print(f"Expected mean (direct calculation): {series.iloc[2:4].mean()}")
print(f"Rolling mean result: {rolling_mean.iloc[3]}")

# Calculate relative error
expected = series.iloc[2:4].mean()
actual = rolling_mean.iloc[3]
if expected != 0:
    rel_error = abs(actual - expected) / abs(expected)
    print(f"Relative error: {rel_error}")
    print(f"Relative error %: {rel_error * 100:.2f}%")

# Check if the mean is within bounds
window_min = series.iloc[2:4].min()
window_max = series.iloc[2:4].max()
print(f"\nWindow bounds check:")
print(f"  Window min: {window_min}")
print(f"  Window max: {window_max}")
print(f"  Rolling mean: {actual}")
print(f"  Is mean within bounds? {window_min <= actual <= window_max}")