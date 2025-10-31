import pandas as pd
import inspect

# Check what pandas is using for JSON
from pandas.io.json._json import ujson_loads, ujson_dumps

print("Checking pandas JSON functions...")
print(f"ujson_loads: {ujson_loads}")
print(f"ujson_dumps: {ujson_dumps}")

# Test these functions directly
test_val = -9223372036854775809
print(f"\nTesting with value {test_val} (outside int64 range):")

# Test dumps
try:
    json_str = ujson_dumps(test_val)
    print(f"ujson_dumps succeeded: {json_str}")
except Exception as e:
    print(f"ujson_dumps error: {e}")

# Test loads with the problematic value
json_str = "-9223372036854775809"
print(f"\nTrying to load JSON string: {json_str}")
try:
    result = ujson_loads(json_str)
    print(f"ujson_loads succeeded: {result}")
except Exception as e:
    print(f"ujson_loads error: {e}")

# Test with maximum int64
print(f"\nTesting with int64 max (9223372036854775807):")
json_str = "9223372036854775807"
try:
    result = ujson_loads(json_str)
    print(f"ujson_loads succeeded: {result}")
except Exception as e:
    print(f"ujson_loads error: {e}")

# Test with minimum int64
print(f"\nTesting with int64 min (-9223372036854775808):")
json_str = "-9223372036854775808"
try:
    result = ujson_loads(json_str)
    print(f"ujson_loads succeeded: {result}")
except Exception as e:
    print(f"ujson_loads error: {e}")