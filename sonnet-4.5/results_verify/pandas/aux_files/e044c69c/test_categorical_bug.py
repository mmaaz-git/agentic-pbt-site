import pandas as pd
from pandas.api.interchange import from_dataframe
import numpy as np

print("Testing categorical null handling in pandas interchange protocol")
print("=" * 60)

# Test 1: Basic reproduction
print("\nTest 1: Basic reproduction")
cat = pd.Categorical.from_codes([-1], categories=['A'])
df = pd.DataFrame({'col': cat})

print(f"Original DataFrame:")
print(df)
print(f"Original value at index 0: {df['col'].iloc[0]}")
print(f"Original value is null: {pd.isna(df['col'].iloc[0])}")

interchange_obj = df.__dataframe__()
result = from_dataframe(interchange_obj)

print(f"\nResult DataFrame:")
print(result)
print(f"Result value at index 0: {result['col'].iloc[0]}")
print(f"Result value is null: {pd.isna(result['col'].iloc[0])}")

# Test 2: Multiple nulls
print("\n" + "=" * 60)
print("\nTest 2: Multiple null values")
cat = pd.Categorical.from_codes([-1, 0, -1, 1, -1], categories=['A', 'B', 'C'])
df = pd.DataFrame({'col': cat})

print(f"Original DataFrame:")
print(df)
for i in range(len(df)):
    print(f"  Index {i}: value={df['col'].iloc[i]}, is_null={pd.isna(df['col'].iloc[i])}")

interchange_obj = df.__dataframe__()
result = from_dataframe(interchange_obj)

print(f"\nResult DataFrame:")
print(result)
for i in range(len(result)):
    print(f"  Index {i}: value={result['col'].iloc[i]}, is_null={pd.isna(result['col'].iloc[i])}")

# Test 3: Check if nulls match
print("\n" + "=" * 60)
print("\nTest 3: Checking null preservation")
mismatches = []
for i in range(len(df)):
    orig_is_null = pd.isna(df['col'].iloc[i])
    result_is_null = pd.isna(result['col'].iloc[i])
    if orig_is_null != result_is_null:
        mismatches.append(i)
        print(f"MISMATCH at index {i}:")
        print(f"  Original: is_null={orig_is_null}, value={df['col'].iloc[i]}")
        print(f"  Result:   is_null={result_is_null}, value={result['col'].iloc[i]}")

if mismatches:
    print(f"\nFOUND {len(mismatches)} mismatches in null handling!")
else:
    print("\nAll nulls preserved correctly.")

# Test 4: Show the modulo wrapping issue
print("\n" + "=" * 60)
print("\nTest 4: Demonstrating modulo wrapping issue")
print(f"-1 % 3 = {-1 % 3}  # This shows why -1 becomes index 0 with modulo")
print("Code -1 should represent null, not category at index 0")