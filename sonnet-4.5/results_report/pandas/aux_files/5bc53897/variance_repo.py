import pandas as pd

data = [5897791891.464727, -2692142700.7497644, 0.0, 1.0]
s = pd.Series(data)
result = s.rolling(window=2).var()

print("Rolling variance results:")
print(result)
print()

print(f"At index 3 (window [0.0, 1.0]): {result.iloc[3]}")
print(f"Expected variance for [0.0, 1.0]: 0.5")
print(f"Actual variance: {result.iloc[3]}")
print()

# Verify that this is indeed negative
if result.iloc[3] < 0:
    print(f"ERROR: Variance is negative ({result.iloc[3]}), which is mathematically impossible!")