import pandas as pd
from io import StringIO

# Test the bug as described in the report
value = -9_223_372_036_854_775_809  # One less than the minimum int64 value
s = pd.Series([value])
print(f"Original Series: {s.tolist()}, dtype={s.dtype}")

json_str = s.to_json(orient='split')
print(f"JSON: {json_str}")

try:
    result = pd.read_json(StringIO(json_str), typ='series', orient='split')
    print(f"Result: {result.tolist()}, dtype={result.dtype}")
except ValueError as e:
    print(f"ValueError raised: {e}")

print("\n" + "="*50 + "\n")

# Test with value just above int64 max
value2 = 9_223_372_036_854_775_808  # One more than the maximum int64 value
s2 = pd.Series([value2])
print(f"Original Series: {s2.tolist()}, dtype={s2.dtype}")

json_str2 = s2.to_json(orient='split')
print(f"JSON: {json_str2}")

try:
    result2 = pd.read_json(StringIO(json_str2), typ='series', orient='split')
    print(f"Result: {result2.tolist()}, dtype={result2.dtype}")
except ValueError as e:
    print(f"ValueError raised: {e}")

print("\n" + "="*50 + "\n")

# Test with value within int64 range
value3 = -9_223_372_036_854_775_808  # Exactly at the minimum int64 value
s3 = pd.Series([value3])
print(f"Original Series (within int64 range): {s3.tolist()}, dtype={s3.dtype}")

json_str3 = s3.to_json(orient='split')
print(f"JSON: {json_str3}")

try:
    result3 = pd.read_json(StringIO(json_str3), typ='series', orient='split')
    print(f"Result: {result3.tolist()}, dtype={result3.dtype}")
    pd.testing.assert_series_equal(result3, s3)
    print("SUCCESS: Round-trip works for int64 boundary value")
except Exception as e:
    print(f"Exception raised: {e}")