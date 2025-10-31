from pandas.io.json import ujson_dumps, ujson_loads
import json

# Test value: -2^64 (-18,446,744,073,709,551,616)
value = -18_446_744_073_709_551_616

print("Testing ujson with -2^64:")
print(f"Original value:     {value}")

# Serialize with ujson
serialized = ujson_dumps(value)
print(f"ujson serialized:   {serialized}")

# Deserialize with ujson
deserialized = ujson_loads(serialized)
print(f"ujson deserialized: {deserialized}")

print(f"Values match:       {deserialized == value}")
print()

# Compare with standard library json
print("Testing stdlib json with -2^64:")
std_serialized = json.dumps(value)
std_deserialized = json.loads(std_serialized)
print(f"stdlib json deserialized: {std_deserialized}")
print(f"stdlib values match:      {std_deserialized == value}")
print()

# Test dictionary round-trip (as in the original Hypothesis test)
print("Testing dictionary round-trip:")
test_dict = {'0': value}
print(f"Original dict:      {test_dict}")

dict_serialized = ujson_dumps(test_dict)
print(f"ujson serialized:   {dict_serialized}")

dict_deserialized = ujson_loads(dict_serialized)
print(f"ujson deserialized: {dict_deserialized}")

print(f"Dicts match:        {dict_deserialized == test_dict}")