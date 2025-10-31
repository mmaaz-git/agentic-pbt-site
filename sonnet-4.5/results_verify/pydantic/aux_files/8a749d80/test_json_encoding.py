import json
import warnings
from pydantic.deprecated.parse import load_str_bytes, Protocol

warnings.filterwarnings('ignore', category=DeprecationWarning)

# Test JSON protocol with encoding parameter
test_data = {"key": "value"}
json_str = json.dumps(test_data)

# Convert to bytes with latin1 encoding
json_bytes = json_str.encode('latin1')
print(f"JSON as latin1 bytes: {json_bytes}")

# Test that JSON protocol respects the encoding parameter when given bytes
result = load_str_bytes(json_bytes, proto=Protocol.json, encoding='latin1')
print(f"JSON decoded with latin1: {result}")

# Verify it works with UTF-8 as well
json_bytes_utf8 = json_str.encode('utf-8')
result_utf8 = load_str_bytes(json_bytes_utf8, proto=Protocol.json, encoding='utf-8')
print(f"JSON decoded with utf-8: {result_utf8}")

# Test with wrong encoding (should fail or produce wrong result)
try:
    result_wrong = load_str_bytes(json_bytes, proto=Protocol.json, encoding='utf-8')
    print(f"JSON decoded with wrong encoding (utf-8 instead of latin1): {result_wrong}")
except Exception as e:
    print(f"JSON with wrong encoding failed: {e}")