import pandas as pd

# Create an empty DataFrame with specific dtypes
df_empty = pd.DataFrame({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='float64')})
print("Original DataFrame dtypes:")
print(df_empty.dtypes)
print()

# Test with orient='tight'
tight_dict = df_empty.to_dict(orient='tight')
reconstructed_tight = pd.DataFrame.from_dict(tight_dict, orient='tight')
print("Reconstructed from 'tight' dtypes:")
print(reconstructed_tight.dtypes)
print()

# Compare dtypes
print("Dtypes match?", df_empty.dtypes.equals(reconstructed_tight.dtypes))
print()

# Check exact dtype comparison
for col in df_empty.columns:
    print(f"Column '{col}':")
    print(f"  Original dtype: {df_empty[col].dtype}")
    print(f"  Reconstructed dtype: {reconstructed_tight[col].dtype}")
    print(f"  Match: {df_empty[col].dtype == reconstructed_tight[col].dtype}")