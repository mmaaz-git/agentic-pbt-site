import pandas as pd
from io import StringIO
from pandas.testing import assert_frame_equal

# Test 1: Empty DataFrame with single column
df1 = pd.DataFrame({"col_0": []})
json_str1 = df1.to_json(orient='split')
df1_roundtrip = pd.read_json(StringIO(json_str1), orient='split')

print("Test 1: Empty DataFrame with single column")
print(f"  Original index: {df1.index} (dtype: {df1.index.dtype})")
print(f"  Roundtrip index: {df1_roundtrip.index} (dtype: {df1_roundtrip.index.dtype})")
print(f"  Match: {df1.index.dtype == df1_roundtrip.index.dtype}\n")

# Test 2: Empty DataFrame with multiple columns
df2 = pd.DataFrame({"col_A": [], "col_B": [], "col_C": []})
json_str2 = df2.to_json(orient='split')
df2_roundtrip = pd.read_json(StringIO(json_str2), orient='split')

print("Test 2: Empty DataFrame with multiple columns")
print(f"  Original index: {df2.index} (dtype: {df2.index.dtype})")
print(f"  Roundtrip index: {df2_roundtrip.index} (dtype: {df2_roundtrip.index.dtype})")
print(f"  Match: {df2.index.dtype == df2_roundtrip.index.dtype}\n")

# Test 3: Non-empty DataFrame (should work fine)
df3 = pd.DataFrame({"col_0": [1, 2, 3]})
json_str3 = df3.to_json(orient='split')
df3_roundtrip = pd.read_json(StringIO(json_str3), orient='split')

print("Test 3: Non-empty DataFrame")
print(f"  Original index: {df3.index} (dtype: {df3.index.dtype})")
print(f"  Roundtrip index: {df3_roundtrip.index} (dtype: {df3_roundtrip.index.dtype})")
print(f"  Match: {df3.index.dtype == df3_roundtrip.index.dtype}\n")

# Test if assert_frame_equal fails
print("Testing assert_frame_equal:")
try:
    assert_frame_equal(df1, df1_roundtrip)
    print("  Test 1 passed with default check_dtype")
except AssertionError as e:
    print(f"  Test 1 failed: {str(e)[:100]}...")

try:
    assert_frame_equal(df1, df1_roundtrip, check_dtype=False)
    print("  Test 1 passed with check_dtype=False")
except AssertionError as e:
    print(f"  Test 1 failed even with check_dtype=False: {str(e)[:100]}...")

try:
    assert_frame_equal(df3, df3_roundtrip)
    print("  Test 3 (non-empty) passed")
except AssertionError as e:
    print(f"  Test 3 (non-empty) failed: {str(e)[:100]}...")