import pandas as pd
import numpy as np

values = [0.0, 1.1125369292536007e-308]
x = pd.Series(values)

print(f"Input values: {x.tolist()}")
print(f"Input non-null count: {x.notna().sum()}")

result = pd.cut(x, bins=2)

print(f"Result: {result.tolist()}")
print(f"Result non-null count: {result.notna().sum()}")
print(f"Expected: 2 non-null values, Got: {result.notna().sum()}")

result, bins = pd.cut(x, bins=2, retbins=True)
print(f"\nBins computed: {bins}")
print(f"Result categories: {result.cat.categories}")

# Additional debugging info
print(f"\nDebug: x.min() = {x.min()}")
print(f"Debug: x.max() = {x.max()}")
print(f"Debug: Range = {x.max() - x.min()}")