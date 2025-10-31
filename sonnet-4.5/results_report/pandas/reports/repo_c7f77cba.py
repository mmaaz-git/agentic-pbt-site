import pandas as pd

df = pd.DataFrame({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='float64')})
print("Original DataFrame:")
print("  Columns:", df.columns.tolist())
print("  Dtypes:", df.dtypes.to_dict())
print("  Shape:", df.shape)
print()

tight_dict = df.to_dict(orient='tight')
print("Dictionary representation (orient='tight'):")
print("  Keys:", list(tight_dict.keys()))
print("  Dict:", tight_dict)
print()

reconstructed = pd.DataFrame.from_dict(tight_dict, orient='tight')
print("Reconstructed DataFrame:")
print("  Columns:", reconstructed.columns.tolist())
print("  Dtypes:", reconstructed.dtypes.to_dict())
print("  Shape:", reconstructed.shape)
print()

print("DataFrames equal?", df.equals(reconstructed))
print()

if not df.equals(reconstructed):
    print("ERROR: Round-trip through to_dict/from_dict with orient='tight' lost dtype information!")
    print("  Expected dtypes:", df.dtypes.to_dict())
    print("  Got dtypes:", reconstructed.dtypes.to_dict())