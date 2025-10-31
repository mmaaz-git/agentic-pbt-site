import json

# Test Python's standard json module with the same float value
value = 1.5932223682757467
print(f"Original value: {value:.17f}")

# Serialize with Python's json
json_str = json.dumps({"value": value})
print(f"JSON string: {json_str}")

# Deserialize
restored = json.loads(json_str)["value"]
print(f"Restored value: {restored:.17f}")
print(f"Match: {value == restored}")

# Check the actual number of digits preserved
print(f"\nActual JSON representation: {json.dumps(value)}")