#!/usr/bin/env python3

# Demonstrate Python behavior with duplicate dict keys
print("Testing Python's behavior with duplicate dictionary keys:")

# Simple example
d = {
    "key1": "value1",
    "key2": "value2",
    "key1": "value3",  # Duplicate key
}

print(f"\nDictionary with duplicate 'key1': {d}")
print(f"Number of keys: {len(d)}")
print(f"Value for 'key1': {d['key1']}")

# Example with function calls as keys
class MyClass:
    def method(self):
        return "method_key"

obj = MyClass()

d2 = {
    obj.method(): "first_value",
    obj.method(): "second_value",  # Same key
}

print(f"\nDictionary with duplicate method calls as keys: {d2}")
print(f"Number of keys: {len(d2)}")

# Demonstrate that this is indeed what's happening in pandas
print("\n" + "="*50)
print("This is exactly what happens in _arrow_dtype_mapping():")
print("The duplicate pa.string() key on line 44 overwrites line 41,")
print("but since both map to pd.StringDtype(), the behavior is unchanged.")
print("However, line 44 is dead code that should be removed.")