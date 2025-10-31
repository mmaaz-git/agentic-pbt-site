import pandas as pd
import json
import io

# Test 1: Value below int64_min
print("Test 1: Value below int64_min")
int64_min = -2**63
value_below_min = int64_min - 1
print(f"Testing with value: {value_below_min}")

data = [{'key': value_below_min}]
json_str = json.dumps(data)
print(f"JSON string: {json_str}")
json_io = io.BytesIO(json_str.encode('utf-8'))

try:
    result = pd.read_json(json_io, lines=False)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test 2: Value above int64_max
print("Test 2: Value above int64_max")
int64_max = 2**63 - 1
value_above_max = int64_max + 1
print(f"Testing with value: {value_above_max}")

data = [{'key': value_above_max}]
json_str = json.dumps(data)
print(f"JSON string: {json_str}")
json_io = io.BytesIO(json_str.encode('utf-8'))

try:
    result = pd.read_json(json_io, lines=False)
    print(f"Success! Result:\n{result}")
    print(f"Data type of the value: {result['key'].dtype}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")