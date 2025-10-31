import io
import pandas as pd
from pandas import Series

s = Series([0.0], dtype='float64')
print(f"Original dtype: {s.dtype}")
print(f"Original values: {s.values}")

json_str = s.to_json(orient='index')
print(f"JSON string: {json_str}")

result = pd.read_json(io.StringIO(json_str), typ='series', orient='index')
print(f"Result dtype: {result.dtype}")
print(f"Result values: {result.values}")

print(f"Bug present: {s.dtype != result.dtype}")

# Test with different values to understand the pattern
print("\n--- Testing with different values ---")

test_cases = [
    [0.0],
    [1.0],
    [1.5],
    [0.0, 1.0],
    [0.5, 1.5],
    [1.0, 2.0, 3.0]
]

for values in test_cases:
    s_test = Series(values, dtype='float64')
    json_test = s_test.to_json(orient='index')
    result_test = pd.read_json(io.StringIO(json_test), typ='series', orient='index')
    print(f"Values: {values} | Original: {s_test.dtype} | After round-trip: {result_test.dtype} | Changed: {s_test.dtype != result_test.dtype}")