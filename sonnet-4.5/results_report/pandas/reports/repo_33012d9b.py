import pandas as pd
from pandas.api.interchange import from_dataframe

# Create a DataFrame with categorical column containing null values
df = pd.DataFrame({"cat_col": pd.Categorical(['a', None])})
print("Original DataFrame:")
print(f"  Values: {list(df['cat_col'])}")
print(f"  Is null: {list(df['cat_col'].isna())}")

# Convert through interchange protocol
interchange_df = df.__dataframe__()
result_df = from_dataframe(interchange_df)

print("\nAfter round-trip through interchange protocol:")
print(f"  Values: {list(result_df['cat_col'])}")
print(f"  Is null: {list(result_df['cat_col'].isna())}")

# Show that null has been converted to actual category
print("\nBug demonstrated:")
print(f"  Original second value is null: {pd.isna(df['cat_col'].iloc[1])}")
print(f"  Result second value is null: {pd.isna(result_df['cat_col'].iloc[1])}")
print(f"  Result second value: '{result_df['cat_col'].iloc[1]}'")

# Additional test with multiple nulls and categories
print("\n" + "="*60)
print("Testing with multiple categories and nulls:")
df2 = pd.DataFrame({"cat_col": pd.Categorical(['a', 'b', None, 'c', None])})
print(f"Original: {list(df2['cat_col'])}")

interchange_df2 = df2.__dataframe__()
result_df2 = from_dataframe(interchange_df2)
print(f"After round-trip: {list(result_df2['cat_col'])}")