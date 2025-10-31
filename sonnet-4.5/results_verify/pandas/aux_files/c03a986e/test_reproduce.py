import pandas as pd

# Create an empty DataFrame with columns
df = pd.DataFrame({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='float64')})
print("Original DataFrame:")
print(df)
print("Original columns:", df.columns.tolist())
print("Original dtypes:")
print(df.dtypes)
print("Number of rows:", len(df))
print()

# Convert to dict with orient='index'
index_dict = df.to_dict(orient='index')
print("to_dict(orient='index') result:", index_dict)
print("Type:", type(index_dict))
print()

# Reconstruct from dict
reconstructed = pd.DataFrame.from_dict(index_dict, orient='index')
print("Reconstructed DataFrame:")
print(reconstructed)
print("Reconstructed columns:", reconstructed.columns.tolist())
print("Reconstructed dtypes:")
print(reconstructed.dtypes if len(reconstructed.columns) > 0 else "No columns")
print("Number of rows:", len(reconstructed))
print()

# Check equality
print("Are DataFrames equal?", df.equals(reconstructed))
print()

# Compare column names
print("Column comparison:")
print("  Original columns:", df.columns.tolist())
print("  Reconstructed columns:", reconstructed.columns.tolist())
print("  Columns match:", df.columns.tolist() == reconstructed.columns.tolist())