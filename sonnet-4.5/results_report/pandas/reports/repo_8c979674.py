import pandas as pd
import numpy as np

series = [0.0, 494699.5, 0.0, 0.0, 0.00390625]
s = pd.Series(series)

rolling_var = s.rolling(window=3).var()
negative_var = rolling_var.iloc[4]

print(f"Rolling variance at index 4: {negative_var}")
print(f"Window data: {series[2:5]}")
print(f"Expected (numpy): {np.var(series[2:5], ddof=1)}")
print(f"BUG: Variance is negative: {negative_var < 0}")

# Additional validation
print(f"\nAll rolling variance values:")
for i, val in enumerate(rolling_var):
    if not pd.isna(val):
        print(f"  Index {i}: {val}")

# Manual calculation to verify
window_data = series[2:5]
mean = sum(window_data) / len(window_data)
squared_diffs = [(x - mean)**2 for x in window_data]
manual_var = sum(squared_diffs) / (len(window_data) - 1)  # ddof=1
print(f"\nManual calculation:")
print(f"  Mean: {mean}")
print(f"  Squared differences: {squared_diffs}")
print(f"  Variance (ddof=1): {manual_var}")