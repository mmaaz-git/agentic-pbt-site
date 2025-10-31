from fastapi.encoders import jsonable_encoder

# Trace through the issue step by step

print("Step 1: Original object with tuple key")
obj = {(1, 2): "value"}
print(f"  obj = {obj}")
print(f"  key type: {type(list(obj.keys())[0])}")

print("\nStep 2: What happens when encoding a tuple directly")
tuple_key = (1, 2)
encoded_tuple = jsonable_encoder(tuple_key)
print(f"  jsonable_encoder((1, 2)) = {encoded_tuple}")
print(f"  type: {type(encoded_tuple)}")

print("\nStep 3: Lists cannot be dictionary keys")
try:
    test_dict = {}
    test_dict[[1, 2]] = "value"
except TypeError as e:
    print(f"  {[1, 2]} as key raises: {e}")

print("\nStep 4: The issue in jsonable_encoder")
print("  The function recursively encodes dict keys (line 281-288)")
print("  Tuples are encoded to lists (line 299-315)")
print("  Then tries to use the list as a key (line 297): encoded_dict[encoded_key] = encoded_value")
print("  This fails because lists are unhashable")

print("\nStep 5: What JSON actually needs")
import json
print("  JSON only supports string keys anyway:")
print(f"  json.dumps({{1: 'value'}}) = {json.dumps({1: 'value'})}")
print(f"  json.dumps({{3.14: 'value'}}) = {json.dumps({3.14: 'value'})}")
print(f"  json.dumps({{True: 'value'}}) = {json.dumps({True: 'value'})}")