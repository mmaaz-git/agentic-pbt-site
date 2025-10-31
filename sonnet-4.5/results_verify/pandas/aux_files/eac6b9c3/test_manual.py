from io import StringIO
import pandas as pd

s = pd.Series([''])
print(f"Original Series:\n{s}")
print(f"Original dtype: {s.dtype}")
print(f"Original value: {s.iloc[0]!r}")

json_str = s.to_json(orient='split')
print(f"\nJSON: {json_str}")

s_recovered = pd.read_json(StringIO(json_str), typ='series', orient='split')
print(f"\nRecovered Series:\n{s_recovered}")
print(f"Recovered dtype: {s_recovered.dtype}")
print(f"Recovered value: {s_recovered.iloc[0]!r}")

# Test workaround
print("\n--- Testing with convert_dates=False ---")
s_recovered_no_convert = pd.read_json(StringIO(json_str), typ='series', orient='split', convert_dates=False)
print(f"Recovered Series (no date conversion):\n{s_recovered_no_convert}")
print(f"Recovered dtype: {s_recovered_no_convert.dtype}")
print(f"Recovered value: {s_recovered_no_convert.iloc[0]!r}")