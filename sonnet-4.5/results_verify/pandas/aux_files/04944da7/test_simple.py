import pandas as pd

# Create empty DataFrame with specific dtypes
df = pd.DataFrame({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='float64')})
print("Original DataFrame:")
print(df)
print("Original dtypes:", df.dtypes.to_dict())
print()

# Convert to dict with orient='tight'
tight_dict = df.to_dict(orient='tight')
print("Dictionary representation with orient='tight':")
print(tight_dict)
print()

# Reconstruct from dict
reconstructed = pd.DataFrame.from_dict(tight_dict, orient='tight')
print("Reconstructed DataFrame:")
print(reconstructed)
print("Reconstructed dtypes:", reconstructed.dtypes.to_dict())
print()

print("Equal?", df.equals(reconstructed))

# Additional test - let's also check what happens with a non-empty DataFrame
print("\n" + "="*50)
print("Testing with non-empty DataFrame:")
df_nonempty = pd.DataFrame({'a': pd.Series([1, 2], dtype='int64'), 'b': pd.Series([3.5, 4.5], dtype='float64')})
print("Original dtypes:", df_nonempty.dtypes.to_dict())
tight_dict_nonempty = df_nonempty.to_dict(orient='tight')
reconstructed_nonempty = pd.DataFrame.from_dict(tight_dict_nonempty, orient='tight')
print("Reconstructed dtypes:", reconstructed_nonempty.dtypes.to_dict())
print("Equal?", df_nonempty.equals(reconstructed_nonempty))