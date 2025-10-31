import pandas as pd
from pandas.api.interchange import from_dataframe

# Create a DataFrame with categorical column containing null
df = pd.DataFrame({"col": pd.Categorical(["cat1", None])})
print("Original DataFrame:")
print(df)
print("\nOriginal values as list:")
print(df["col"].tolist())

# Convert through the interchange protocol
result = from_dataframe(df.__dataframe__())
print("\nDataFrame after round-trip:")
print(result)
print("\nValues after round-trip as list:")
print(result["col"].tolist())

# Check if values match
print("\nComparison:")
print(f"Original: {df['col'].tolist()}")
print(f"After round-trip: {result['col'].tolist()}")
print(f"Are they equal? {df['col'].tolist() == result['col'].tolist()}")