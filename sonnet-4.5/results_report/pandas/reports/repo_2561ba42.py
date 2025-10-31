import pandas as pd

data = [1.0, -4294967297.0, 0.99999, 0.0, 0.0, 1.6675355247098508e-21]
s = pd.Series(data)
rolling_mean = s.rolling(window=3).mean()
rolling_min = s.rolling(window=3).min()
rolling_max = s.rolling(window=3).max()

window_at_5 = data[3:6]
expected_mean = sum(window_at_5) / 3
actual_mean = rolling_mean.iloc[5]

print(f"Data: {data}")
print(f"\nWindow at index 5: {window_at_5}")
print(f"Expected mean: {expected_mean}")
print(f"Pandas rolling mean: {actual_mean}")
print(f"Pandas rolling min: {rolling_min.iloc[5]}")
print(f"Pandas rolling max: {rolling_max.iloc[5]}")
print(f"\nBug: mean ({actual_mean}) > max ({max(window_at_5)})")
print(f"Mathematical violation: min <= mean <= max is FALSE")
print(f"Relative error: {abs(actual_mean - expected_mean) / abs(expected_mean) * 100:.2e}%")