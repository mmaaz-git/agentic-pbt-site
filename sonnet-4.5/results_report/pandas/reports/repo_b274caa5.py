import pandas as pd
from pandas.api.interchange import from_dataframe

# Create a DataFrame with categorical data that includes null values
df = pd.DataFrame({
    "cat": pd.Categorical(["a", "b", None, "c"], categories=["a", "b", "c"])
})

print("Original DataFrame:")
print(df)
print(f"Null values: {df['cat'].isna().tolist()}")
print(f"Categories: {df['cat'].cat.categories.tolist()}")

# Convert through the interchange protocol
df_interchange = df.__dataframe__()
df_result = from_dataframe(df_interchange)

print("\nAfter interchange conversion:")
print(df_result)
print(f"Null values: {df_result['cat'].isna().tolist()}")
print(f"Categories: {df_result['cat'].cat.categories.tolist()}")

# Show that the null value was converted to a valid category
print("\nComparison:")
print(f"Original value at index 2: {repr(df['cat'].iloc[2])}")
print(f"Result value at index 2: {repr(df_result['cat'].iloc[2])}")