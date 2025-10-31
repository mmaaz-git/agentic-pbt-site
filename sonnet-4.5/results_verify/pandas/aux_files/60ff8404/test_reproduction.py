import pandas as pd
from pandas.api.interchange import from_dataframe

# Reproducing the bug as described
print("=" * 60)
print("REPRODUCING THE BUG")
print("=" * 60)

df = pd.DataFrame({"col": pd.array([1, None, 3], dtype="Int64")})
print(f"Original dtype: {df['col'].dtype}")
print(f"Original data:\n{df}")

interchange_obj = df.__dataframe__()
result_df = from_dataframe(interchange_obj)
print(f"\nResult dtype: {result_df['col'].dtype}")
print(f"Result data:\n{result_df}")

print(f"\nDtypes match? {df['col'].dtype == result_df['col'].dtype}")

# Check with assertion
try:
    assert df["col"].dtype == result_df["col"].dtype
    print("Assertion passed: dtypes match")
except AssertionError:
    print("AssertionError: Dtype changed from Int64 to float64")

print("\n" + "=" * 60)
print("TESTING WITH [None] only:")
print("=" * 60)

df2 = pd.DataFrame({"col": pd.array([None], dtype="Int64")})
print(f"Original dtype: {df2['col'].dtype}")
print(f"Original data:\n{df2}")

interchange_obj2 = df2.__dataframe__()
result_df2 = from_dataframe(interchange_obj2)
print(f"\nResult dtype: {result_df2['col'].dtype}")
print(f"Result data:\n{result_df2}")

print(f"\nDtypes match? {df2['col'].dtype == result_df2['col'].dtype}")

# Test with assertion
try:
    pd.testing.assert_frame_equal(result_df2, df2)
    print("assert_frame_equal passed")
except AssertionError as e:
    print(f"assert_frame_equal failed: {e}")