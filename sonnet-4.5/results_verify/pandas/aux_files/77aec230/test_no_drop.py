import pandas as pd

print("Testing with drop_first=False...")
df = pd.DataFrame({"A": ["cat"], "B": ["x"]})
print(f"Original DataFrame:\n{df}\n")

dummies = pd.get_dummies(df, drop_first=False, dtype=int)
print(f"Result of get_dummies (drop_first=False):\n{dummies}")
print(f"Shape: {dummies.shape}")
print(f"Columns: {list(dummies.columns)}\n")

try:
    recovered = pd.from_dummies(dummies, sep="_")
    print(f"Recovered DataFrame:\n{recovered}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")