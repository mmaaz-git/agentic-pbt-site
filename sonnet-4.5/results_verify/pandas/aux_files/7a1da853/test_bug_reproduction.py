#!/usr/bin/env python3

from io import StringIO
import pandas as pd
import sys
import json

print("=== System Information ===")
print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"sys.float_info.max: {sys.float_info.max}")
print()

print("=== Test Case from Bug Report ===")
# The specific value from the bug report
test_value = 1.7976931345e+308

print(f"Test value: {test_value}")
print(f"Test value < sys.float_info.max: {test_value < sys.float_info.max}")
print(f"Test value is finite: {test_value != float('inf') and test_value != float('-inf')}")
print()

print("=== Round-trip through pandas to_json/read_json ===")
df = pd.DataFrame([[test_value]])
print(f"Original DataFrame value: {df.iloc[0, 0]}")
print(f"Original value is finite: {df.iloc[0, 0] != float('inf')}")

json_str = df.to_json(orient='columns')
print(f"JSON string: {json_str}")

df_recovered = pd.read_json(StringIO(json_str), orient='columns')
print(f"Recovered DataFrame value: {df_recovered.iloc[0, 0]}")
print(f"Recovered value is infinite: {df_recovered.iloc[0, 0] == float('inf')}")
print(f"Values are equal: {df.iloc[0, 0] == df_recovered.iloc[0, 0]}")
print()

print("=== Testing with Python's stdlib json module ===")
# Test with Python's standard json module
test_dict = {"value": test_value}
json_str_stdlib = json.dumps(test_dict)
print(f"Stdlib JSON string: {json_str_stdlib}")
recovered_stdlib = json.loads(json_str_stdlib)
print(f"Recovered value from stdlib: {recovered_stdlib['value']}")
print(f"Stdlib preserves value: {test_dict['value'] == recovered_stdlib['value']}")
print()

print("=== Testing precise_float parameter ===")
# Test with precise_float=True
df2 = pd.DataFrame([[test_value]])
json_str2 = df2.to_json(orient='columns')
print(f"JSON string (same as before): {json_str2}")
try:
    df_recovered2 = pd.read_json(StringIO(json_str2), orient='columns', precise_float=True)
    print(f"Recovered with precise_float=True: {df_recovered2.iloc[0, 0]}")
    print(f"Is infinite with precise_float=True: {df_recovered2.iloc[0, 0] == float('inf')}")
except Exception as e:
    print(f"Error with precise_float=True: {e}")
print()

print("=== Testing different double_precision values ===")
for precision in [10, 15, 17]:
    try:
        df3 = pd.DataFrame([[test_value]])
        json_str3 = df3.to_json(orient='columns', double_precision=precision)
        print(f"Precision {precision}: JSON = {json_str3}")
        df_recovered3 = pd.read_json(StringIO(json_str3), orient='columns')
        print(f"  Recovered value: {df_recovered3.iloc[0, 0]}, is inf: {df_recovered3.iloc[0, 0] == float('inf')}")
    except Exception as e:
        print(f"Precision {precision}: Error - {e}")
print()

print("=== Testing the parsed JSON value directly ===")
# Let's manually parse the JSON string that pandas produces
json_from_pandas = {"0": {"0": 1.797693135e+308}}
print(f"Manually created JSON dict: {json_from_pandas}")
print(f"Value from dict: {json_from_pandas['0']['0']}")
print(f"Is this value infinite: {json_from_pandas['0']['0'] == float('inf')}")

# Parse the actual string
import ast
parsed = ast.literal_eval('{"0":{"0":1.797693135e+308}}')
print(f"Parsed with ast.literal_eval: {parsed['0']['0']}")
print(f"Is parsed value infinite: {parsed['0']['0'] == float('inf')}")