import pandas as pd
from pandas.api.interchange import from_dataframe

# Test Case 1: Single null value
print("=== Test Case 1: Single null value ===")
cat = pd.Categorical.from_codes([-1], categories=['A'])
df = pd.DataFrame({'col': cat})

print(f"Original DataFrame:")
print(df)
print(f"Original value is null: {pd.isna(df['col'].iloc[0])}")

interchange_obj = df.__dataframe__()
result = from_dataframe(interchange_obj)

print(f"\nAfter interchange conversion:")
print(result)
print(f"Result value is null: {pd.isna(result['col'].iloc[0])}")
print(f"Result value: {result['col'].iloc[0]}")

# Test Case 2: Multiple nulls with 3 categories
print("\n=== Test Case 2: Multiple nulls (3 categories) ===")
cat2 = pd.Categorical.from_codes([-1, 0, -1, 1, -1], categories=['A', 'B', 'C'])
df2 = pd.DataFrame({'col': cat2})

print(f"Original DataFrame:")
print(df2)
print(f"Original nulls at indices: {[i for i in range(len(df2)) if pd.isna(df2['col'].iloc[i])]}")

interchange_obj2 = df2.__dataframe__()
result2 = from_dataframe(interchange_obj2)

print(f"\nAfter interchange conversion:")
print(result2)
print(f"Result nulls at indices: {[i for i in range(len(result2)) if pd.isna(result2['col'].iloc[i])]}")

# Show what -1 % 3 equals
print(f"\n-1 % 3 = {-1 % 3}")
print(f"categories[2] = '{['A', 'B', 'C'][2]}'")