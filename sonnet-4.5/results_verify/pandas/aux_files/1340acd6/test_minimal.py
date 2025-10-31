import pandas as pd

print("Test 1: Basic reproduction case")
print("-" * 40)
values = [0.0, 2.225e-313]
s = pd.Series(values)

print(f"Input values: {values}")
result = pd.cut(s, bins=2)
print(f"Result: {result.tolist()}")
print(f"All values are NaN: {result.isna().all()}")

assert result.isna().all(), "Expected all NaN values"
print("✓ Confirmed: All values returned as NaN\n")