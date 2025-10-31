import pandas as pd

print("Creating DataFrame with single unique values...")
df = pd.DataFrame({"A": ["cat"], "B": ["x"]})
print(f"Original DataFrame:\n{df}\n")

print("Applying get_dummies with drop_first=True...")
dummies = pd.get_dummies(df, drop_first=True, dtype=int)
print(f"Result of get_dummies:\n{dummies}")
print(f"Shape: {dummies.shape}")
print(f"Columns: {list(dummies.columns)}\n")

print("Attempting to recover original with from_dummies...")
default_cats = {"A": "cat", "B": "x"}
print(f"Default categories: {default_cats}")

try:
    recovered = pd.from_dummies(dummies, sep="_", default_category=default_cats)
    print(f"Recovered DataFrame:\n{recovered}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")