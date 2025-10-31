import pandas as pd
import pandas.api.interchange as interchange

# Create a DataFrame with a categorical column containing null values
df = pd.DataFrame({'cat': pd.Categorical(['a', None])})
print("Original DataFrame:")
print(df)
print(f"Type of original column: {type(df['cat'])}")
print(f"Original categories: {df['cat'].cat.categories.tolist()}")
print(f"Original codes: {df['cat'].cat.codes.tolist()}")

# Round-trip through interchange protocol
interchange_obj = df.__dataframe__()
result = interchange.from_dataframe(interchange_obj)

print("\nResult DataFrame after round-trip:")
print(result)
print(f"Type of result column: {type(result['cat'])}")
print(f"Result categories: {result['cat'].cat.categories.tolist()}")
print(f"Result codes: {result['cat'].cat.codes.tolist()}")

# Check if null is preserved
print(f"\nOriginal value at position 1: {df.iloc[1, 0]}")
print(f"Result value at position 1: {result.iloc[1, 0]}")
print(f"Is original value null? {pd.isna(df.iloc[1, 0])}")
print(f"Is result value null? {pd.isna(result.iloc[1, 0])}")

# This assertion should pass but will fail due to the bug
try:
    assert pd.isna(result.iloc[1, 0]), f"Expected null but got {result.iloc[1, 0]}"
    print("\n✓ Assertion passed: Null value was preserved")
except AssertionError as e:
    print(f"\n✗ Assertion failed: {e}")