from pydantic import BaseModel
import json

class MyModel(BaseModel):
    data: bytes

# Test with non-UTF-8 bytes
print("Testing with non-UTF-8 bytes:")
try:
    model = MyModel(data=b'\x80')
    print(f"Model created: {model}")
    print(f"Model data: {model.data!r}")
    json_str = model.model_dump_json()
    print(f"JSON output: {json_str}")
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e).__name__}")

# Test with valid UTF-8 bytes
print("\nTesting with valid UTF-8 bytes:")
try:
    model = MyModel(data=b'Hello, World!')
    print(f"Model created: {model}")
    print(f"Model data: {model.data!r}")
    json_str = model.model_dump_json()
    print(f"JSON output: {json_str}")
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e).__name__}")

# Test manual JSON encoding with Python's json module
print("\nTesting Python's standard json module with bytes:")
try:
    json.dumps(b'\x80')
except TypeError as e:
    print(f"Standard json module error: {e}")
    print("Python's json module doesn't support bytes directly")