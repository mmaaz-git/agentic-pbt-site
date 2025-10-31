import pandas as pd

# Create a minimal DataFrame with single unique values per column
df = pd.DataFrame({"A": ["cat"], "B": ["x"]})
print("Original DataFrame:")
print(df)
print()

# Apply get_dummies with drop_first=True
dummies = pd.get_dummies(df, drop_first=True, dtype=int)
print("Result from get_dummies(drop_first=True):")
print(dummies)
print(f"Shape: {dummies.shape}")
print()

# Try to reconstruct using from_dummies
default_cats = {"A": "cat", "B": "x"}
print(f"Attempting from_dummies with default_category={default_cats}")
print()

try:
    recovered = pd.from_dummies(dummies, sep="_", default_category=default_cats)
    print("Successfully recovered:")
    print(recovered)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")