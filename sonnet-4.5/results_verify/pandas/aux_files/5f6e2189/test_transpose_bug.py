import pandas as pd
import numpy as np
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames

# First, let's run the exact reproduction code from the bug report
print("=" * 60)
print("REPRODUCING BUG AS DESCRIBED IN REPORT")
print("=" * 60)

df = pd.DataFrame({'a': [], 'b': []})
df['a'] = df['a'].astype(int)
df['b'] = df['b'].astype(int)

print("Original dtypes:", df.dtypes.to_dict())
print("Shape:", df.shape)
print("DataFrame content:")
print(df)

# First transpose
df_t = df.T
print("\nAfter first transpose (df.T):")
print("Shape:", df_t.shape)
print("Columns:", df_t.columns.tolist())
print("Index:", df_t.index.tolist())
print("DataFrame content:")
print(df_t)

# Second transpose
result = df.T.T

print("\nAfter df.T.T dtypes:", result.dtypes.to_dict())
print("Shape:", result.shape)
print("Bug: int64 columns became", result['a'].dtype)
print("DataFrame content:")
print(result)

# Check equality
print("\nChecking equality:")
print("df.equals(df.T.T):", df.equals(result))
try:
    pd.testing.assert_frame_equal(df, result)
    print("pd.testing.assert_frame_equal: PASSED")
except AssertionError as e:
    print("pd.testing.assert_frame_equal: FAILED")
    print("Error:", str(e))

print("\n" + "=" * 60)
print("RUNNING PROPERTY-BASED TEST")
print("=" * 60)

# Now run the property-based test
@given(data_frames([
    column('a', dtype=int),
    column('b', dtype=int),
]))
@settings(max_examples=200)
def test_transpose_transpose_identity(df):
    result = df.T.T
    pd.testing.assert_frame_equal(result, df)

try:
    test_transpose_transpose_identity()
    print("Property-based test PASSED")
except Exception as e:
    print("Property-based test FAILED")
    print("Error:", str(e))

print("\n" + "=" * 60)
print("ADDITIONAL TEST CASES")
print("=" * 60)

# Test with non-empty DataFrame
print("\n1. Non-empty integer DataFrame:")
df_nonempty = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
print("Original dtypes:", df_nonempty.dtypes.to_dict())
result_nonempty = df_nonempty.T.T
print("After df.T.T dtypes:", result_nonempty.dtypes.to_dict())
print("Equality:", df_nonempty.equals(result_nonempty))

# Test with empty float DataFrame
print("\n2. Empty float DataFrame:")
df_float = pd.DataFrame({'a': [], 'b': []})
df_float['a'] = df_float['a'].astype(float)
df_float['b'] = df_float['b'].astype(float)
print("Original dtypes:", df_float.dtypes.to_dict())
result_float = df_float.T.T
print("After df.T.T dtypes:", result_float.dtypes.to_dict())
print("Equality:", df_float.equals(result_float))

# Test with mixed dtypes
print("\n3. Empty mixed dtype DataFrame:")
df_mixed = pd.DataFrame({'a': [], 'b': []})
df_mixed['a'] = df_mixed['a'].astype(int)
df_mixed['b'] = df_mixed['b'].astype(float)
print("Original dtypes:", df_mixed.dtypes.to_dict())
result_mixed = df_mixed.T.T
print("After df.T.T dtypes:", result_mixed.dtypes.to_dict())

# Test with single column
print("\n4. Empty single column DataFrame:")
df_single = pd.DataFrame({'a': []})
df_single['a'] = df_single['a'].astype(int)
print("Original dtypes:", df_single.dtypes.to_dict())
result_single = df_single.T.T
print("After df.T.T dtypes:", result_single.dtypes.to_dict())

# Understand what happens during transpose
print("\n" + "=" * 60)
print("ANALYZING TRANSPOSE BEHAVIOR")
print("=" * 60)

df = pd.DataFrame({'a': [], 'b': []})
df['a'] = df['a'].astype(int)
df['b'] = df['b'].astype(int)

print("Original DataFrame:")
print("  Shape:", df.shape)
print("  Dtypes:", df.dtypes.to_dict())
print("  Columns:", df.columns.tolist())
print("  Index:", df.index.tolist())

df_t = df.T
print("\nAfter first transpose (df.T):")
print("  Shape:", df_t.shape)
print("  Dtypes:", df_t.dtypes.to_dict() if len(df_t.columns) > 0 else "No columns")
print("  Columns:", df_t.columns.tolist())
print("  Index:", df_t.index.tolist())

df_tt = df_t.T
print("\nAfter second transpose (df.T.T):")
print("  Shape:", df_tt.shape)
print("  Dtypes:", df_tt.dtypes.to_dict())
print("  Columns:", df_tt.columns.tolist())
print("  Index:", df_tt.index.tolist())