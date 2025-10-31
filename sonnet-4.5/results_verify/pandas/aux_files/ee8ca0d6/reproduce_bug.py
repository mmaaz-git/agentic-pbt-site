import pandas as pd
from pandas.io.json import read_json, to_json
import io

df = pd.DataFrame([])
print(f"Original index: {df.index}")
print(f"Original index type: {type(df.index).__name__}")
print(f"Original index dtype: {df.index.dtype}")

json_str = to_json(None, df, orient='split')
print(f"\nJSON: {json_str}")

result = read_json(io.StringIO(json_str), orient='split')
print(f"\nResult index: {result.index}")
print(f"Result index type: {type(result.index).__name__}")
print(f"Result index dtype: {result.index.dtype}")

# Also check inferred_type
print(f"\nOriginal index inferred_type: {df.index.inferred_type}")
print(f"Result index inferred_type: {result.index.inferred_type}")

# Check if they're equal
try:
    pd.testing.assert_frame_equal(result, df)
    print("\nDataFrames are equal")
except AssertionError as e:
    print(f"\nDataFrames are NOT equal: {e}")