import pandas as pd
from pandas.io.json import read_json, to_json
import io

# Test with orient='table'
df = pd.DataFrame([])
print("Testing with orient='table':")
print(f"Original index: {df.index}")
print(f"Original index type: {type(df.index).__name__}")
print(f"Original index dtype: {df.index.dtype}")

json_str = to_json(None, df, orient='table', date_format='iso')
print(f"\nJSON: {json_str}")

result = read_json(io.StringIO(json_str), orient='table')
print(f"\nResult index: {result.index}")
print(f"Result index type: {type(result.index).__name__}")
print(f"Result index dtype: {result.index.dtype}")

try:
    pd.testing.assert_frame_equal(result, df)
    print("\nDataFrames are equal with orient='table'")
except AssertionError as e:
    print(f"\nDataFrames are NOT equal: {e}")

# Also test with non-empty DataFrame with orient='split'
print("\n" + "="*50)
print("Testing non-empty DataFrame with orient='split':")
df2 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})
print(f"Original index: {df2.index}")
print(f"Original index type: {type(df2.index).__name__}")

json_str2 = to_json(None, df2, orient='split')
result2 = read_json(io.StringIO(json_str2), orient='split')
print(f"Result index: {result2.index}")
print(f"Result index type: {type(result2.index).__name__}")

try:
    pd.testing.assert_frame_equal(result2, df2)
    print("Non-empty DataFrames are equal with orient='split'")
except AssertionError as e:
    print(f"Non-empty DataFrames are NOT equal: {e}")