import pandas as pd
import numpy as np
from pandas.api.interchange import from_dataframe

# Create a categorical with missing values
categories = ['a', 'b', 'c']
codes = np.array([0, 1, 2, -1, 0, 1], dtype='int8')  # -1 represents missing value

cat = pd.Categorical.from_codes(codes, categories=categories)
df = pd.DataFrame({'cat': cat})

print("Original DataFrame:")
print(df)
print(f"\nOriginal missing values count: {df.isna().sum().sum()}")
print(f"Original values at each index:")
for i in range(len(df)):
    val = df.iloc[i, 0]
    print(f"  Index {i}: {repr(val)}")

# Convert through interchange protocol
result = from_dataframe(df.__dataframe__())

print("\n\nAfter interchange conversion:")
print(result)
print(f"\nMissing values count after conversion: {result.isna().sum().sum()}")
print(f"Values at each index after conversion:")
for i in range(len(result)):
    val = result.iloc[i, 0]
    print(f"  Index {i}: {repr(val)}")

print("\n\nBUG DEMONSTRATION:")
print(f"Index 3 in original: {repr(df.iloc[3, 0])}")
print(f"Index 3 after conversion: {repr(result.iloc[3, 0])}")
print(f"Expected at index 3: NaN (missing value)")
print(f"Actual at index 3: '{result.iloc[3, 0]}' (category 'c')")
print("\nThe missing value was incorrectly converted to category 'c'!")