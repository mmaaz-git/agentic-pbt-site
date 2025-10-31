#!/usr/bin/env python3
"""Test the proposed fix for DictionarySerializer"""

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'test_settings'

import django
django.setup()

# Create a patched version of DictionarySerializer
from django.db.migrations.serializer import BaseSerializer, serializer_factory

class FixedDictionarySerializer(BaseSerializer):
    """Fixed version that sorts by repr() of keys"""
    def serialize(self):
        imports = set()
        strings = []
        # Use key=lambda item: repr(item[0]) to handle mixed-type keys
        for k, v in sorted(self.value.items(), key=lambda item: repr(item[0])):
            k_string, k_imports = serializer_factory(k).serialize()
            v_string, v_imports = serializer_factory(v).serialize()
            imports.update(k_imports)
            imports.update(v_imports)
            strings.append((k_string, v_string))
        return "{%s}" % (", ".join("%s: %s" % (k, v) for k, v in strings)), imports

print("Testing fixed DictionarySerializer...")
print("=" * 60)

# Test cases
test_cases = [
    {1: 'value1', 'key2': 'value2'},
    {1: 10, 'a': 20},
    {'b': 30, 2: 40, 'a': 50},
    {True: 'bool', 1: 'int', 'str': 'string'},  # Note: True and 1 are same in dict
]

for test_dict in test_cases:
    print(f"\nTest dict: {test_dict}")

    # Create a fixed serializer
    serializer = FixedDictionarySerializer(test_dict)

    try:
        serialized, imports = serializer.serialize()
        print(f"Success! Serialized: {serialized}")

        # Test round-trip
        exec_globals = {}
        for imp in imports:
            exec(imp, exec_globals)
        deserialized = eval(serialized, exec_globals)

        if deserialized == test_dict:
            print(f"✓ Round-trip successful: {deserialized}")
        else:
            print(f"✗ Round-trip failed!")
            print(f"  Original:     {test_dict}")
            print(f"  Deserialized: {deserialized}")

    except Exception as e:
        print(f"✗ Failed: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Testing deterministic order...")

# Check that sorting by repr() gives deterministic results
test_dict = {2: 'two', 'a': 'letter_a', 10: 'ten', 'b': 'letter_b', 1: 'one'}
serializer = FixedDictionarySerializer(test_dict)
serialized1, _ = serializer.serialize()

# Serialize again
serializer = FixedDictionarySerializer(test_dict)
serialized2, _ = serializer.serialize()

print(f"First serialization:  {serialized1}")
print(f"Second serialization: {serialized2}")
if serialized1 == serialized2:
    print("✓ Serialization is deterministic")
else:
    print("✗ Serialization is not deterministic")