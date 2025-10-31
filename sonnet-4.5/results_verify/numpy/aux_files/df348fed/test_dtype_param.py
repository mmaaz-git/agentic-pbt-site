import io
import pandas as pd
from pandas import Series

# Test the dtype=False workaround suggested in the bug report
print("Testing with dtype=False parameter:")
s = Series([0.0, 1.0, 2.0], dtype='float64')
print(f"Original dtype: {s.dtype}")

json_str = s.to_json(orient='index')
print(f"JSON: {json_str}")

# With default dtype (should be True)
result_default = pd.read_json(io.StringIO(json_str), typ='series', orient='index')
print(f"Result with default dtype: {result_default.dtype}")

# With dtype=False
result_false = pd.read_json(io.StringIO(json_str), typ='series', orient='index', dtype=False)
print(f"Result with dtype=False: {result_false.dtype}")

# With dtype=True explicitly
result_true = pd.read_json(io.StringIO(json_str), typ='series', orient='index', dtype=True)
print(f"Result with dtype=True: {result_true.dtype}")

# Test if dtype=False preserves the float64 type
print(f"\ndtype=False preserves float64: {result_false.dtype == s.dtype}")

# Check the actual values
print(f"\nOriginal values: {s.values}")
print(f"Result with dtype=False values: {result_false.values}")
print(f"Values are equal: {(s.values == result_false.values).all()}")