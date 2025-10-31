import json
import pickle
import warnings
from pydantic.deprecated.parse import load_str_bytes, Protocol

warnings.filterwarnings('ignore', category=DeprecationWarning)

print("=" * 60)
print("Testing symmetry for JSON protocol:")
print("=" * 60)

# JSON: bytes -> function -> object
json_data = {"key": "café"}
json_str = json.dumps(json_data, ensure_ascii=False)
json_bytes = json_str.encode('latin1')
print(f"1. Original JSON object: {json_data}")
print(f"2. JSON bytes (latin1): {json_bytes}")
result = load_str_bytes(json_bytes, proto=Protocol.json, encoding='latin1')
print(f"3. Loaded from bytes with encoding='latin1': {result}")
print(f"   Matches original: {result == json_data}")

# JSON: string -> function -> object
result_from_str = load_str_bytes(json_str, proto=Protocol.json, encoding='latin1')
print(f"4. Loaded from string (encoding ignored for strings): {result_from_str}")
print(f"   Matches original: {result_from_str == json_data}")

print("\n" + "=" * 60)
print("Testing symmetry for Pickle protocol:")
print("=" * 60)

# Pickle: bytes -> function -> object
pickle_data = {"key": "café"}
pickle_bytes = pickle.dumps(pickle_data)
print(f"1. Original pickle object: {pickle_data}")
print(f"2. Pickle bytes: {pickle_bytes[:50]}...")  # truncate for readability
result = load_str_bytes(pickle_bytes, proto=Protocol.pickle, encoding='latin1', allow_pickle=True)
print(f"3. Loaded from bytes: {result}")
print(f"   Matches original: {result == pickle_data}")

# Pickle: string -> function -> object
# This is where the bug occurs
# We need to decode the pickle bytes to a string somehow
# The natural way would be to use the same encoding parameter
pickle_str = pickle_bytes.decode('latin1')
print(f"4. Pickle as string (decoded with latin1): {repr(pickle_str[:50])}...")
try:
    result_from_str = load_str_bytes(pickle_str, proto=Protocol.pickle, encoding='latin1', allow_pickle=True)
    print(f"5. Loaded from string with encoding='latin1': {result_from_str}")
    print(f"   Matches original: {result_from_str == pickle_data}")
except Exception as e:
    print(f"5. Failed to load from string: {e}")