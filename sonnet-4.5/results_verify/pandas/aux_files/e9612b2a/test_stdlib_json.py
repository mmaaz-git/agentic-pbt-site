import json

# Test the same values with stdlib json
print("Testing with stdlib json:")

value = -18_446_744_073_709_551_616
print(f"\nTesting value: {value} (-2^64)")
serialized = json.dumps(value)
deserialized = json.loads(serialized)
print(f"Original:     {value}")
print(f"Serialized:   {serialized}")
print(f"Deserialized: {deserialized}")
print(f"Match:        {deserialized == value}")
print(f"Type:         {type(deserialized)}")

# Test with dictionary
d = {'0': -18446744073709551616}
print(f"\nOriginal dict: {d}")
serialized_dict = json.dumps(d)
print(f"Serialized: {serialized_dict}")
deserialized_dict = json.loads(serialized_dict)
print(f"Deserialized: {deserialized_dict}")
print(f"Match: {deserialized_dict == d}")

# Test a range of values
print("\nTesting values around -2^64 with stdlib json:")
test_values = [
    -18_446_744_073_709_551_615,  # -2^64 + 1
    -18_446_744_073_709_551_616,  # -2^64
    -18_446_744_073_709_551_617,  # -2^64 - 1
    -9_223_372_036_854_775_808,   # -2^63
    -9_223_372_036_854_775_809,   # -2^63 - 1
]

for val in test_values:
    try:
        s = json.dumps(val)
        d = json.loads(s)
        print(f"Value {val}: serialized={s}, deserialized={d}, match={d==val}")
    except Exception as e:
        print(f"Value {val}: ERROR - {e}")