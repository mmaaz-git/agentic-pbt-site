import pandas as pd
import numpy as np

# Create a DataFrame with mixed dtypes (float64 and int64)
df = pd.DataFrame({'a': [1.5], 'b': [2]})

print("Original DataFrame:")
print(df)
print("\nOriginal dtypes:")
print(df.dtypes)
print("\nOriginal 'b' column values and dtype:")
print(f"Values: {df['b'].values}")
print(f"Dtype: {df['b'].dtype}")

# Transpose the DataFrame
df_t = df.T
print("\n\nAfter first transpose (df.T):")
print(df_t)
print("\nDtypes after first transpose:")
print(df_t.dtypes)

# Transpose again to get back to original
df_tt = df_t.T
print("\n\nAfter double transpose (df.T.T):")
print(df_tt)
print("\nDtypes after double transpose:")
print(df_tt.dtypes)
print("\n'b' column after double transpose:")
print(f"Values: {df_tt['b'].values}")
print(f"Dtype: {df_tt['b'].dtype}")

# Check if dtypes are preserved
print("\n\nComparison:")
print(f"Original 'b' dtype: {df['b'].dtype}")
print(f"After T.T 'b' dtype: {df_tt['b'].dtype}")
print(f"Are they equal? {df['b'].dtype == df_tt['b'].dtype}")

# This assertion will fail
try:
    assert df['b'].dtype == df_tt['b'].dtype, f"Expected {df['b'].dtype} but got {df_tt['b'].dtype}"
    print("\nAssertion passed: dtypes are preserved")
except AssertionError as e:
    print(f"\nAssertion failed: {e}")