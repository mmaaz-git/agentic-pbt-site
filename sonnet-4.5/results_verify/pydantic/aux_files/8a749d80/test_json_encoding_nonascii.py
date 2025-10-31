import json
import warnings
from pydantic.deprecated.parse import load_str_bytes, Protocol

warnings.filterwarnings('ignore', category=DeprecationWarning)

# Test JSON protocol with non-ASCII characters
test_data = {"key": "café"}  # Contains non-ASCII character é
json_str = json.dumps(test_data, ensure_ascii=False)
print(f"JSON string with non-ASCII: {json_str}")

# Convert to bytes with utf-8 encoding
json_bytes_utf8 = json_str.encode('utf-8')
print(f"JSON as UTF-8 bytes: {json_bytes_utf8}")

# Test that JSON protocol respects the encoding parameter when given bytes
result = load_str_bytes(json_bytes_utf8, proto=Protocol.json, encoding='utf-8')
print(f"JSON decoded with utf-8: {result}")

# Try with wrong encoding (should fail)
try:
    result_wrong = load_str_bytes(json_bytes_utf8, proto=Protocol.json, encoding='latin1')
    print(f"JSON decoded with wrong encoding (latin1 instead of utf-8): {result_wrong}")
except Exception as e:
    print(f"JSON with wrong encoding failed: {e}")