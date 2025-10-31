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

# Let's also manually compute the variance to understand what's happening
window_data = [0.0, 0.0, 0.00390625]
mean = sum(window_data) / len(window_data)
squared_diffs = [(x - mean) ** 2 for x in window_data]
variance_manual = sum(squared_diffs) / (len(window_data) - 1)  # ddof=1

print(f"\nManual calculation:")
print(f"Window: {window_data}")
print(f"Mean: {mean}")
print(f"Squared diffs: {squared_diffs}")
print(f"Variance (ddof=1): {variance_manual}")