#!/usr/bin/env python3
"""Test DictionarySerializer bug with mixed-type keys"""

import sys
import os

# Set up Django
os.environ['DJANGO_SETTINGS_MODULE'] = 'test_settings'

# Create minimal Django settings
with open('test_settings.py', 'w') as f:
    f.write("""
SECRET_KEY = 'test-secret-key'
DEBUG = True
INSTALLED_APPS = []
DATABASES = {}
USE_TZ = False
""")

import django
django.setup()

from django.db.migrations.serializer import serializer_factory

print("Testing DictionarySerializer with mixed-type keys...")
print("=" * 60)

# Test case 1: The failing input from the bug report
test_dict = {1: 'value1', 'key2': 'value2'}
print(f"Test dict: {test_dict}")

try:
    serialized, imports = serializer_factory(test_dict).serialize()
    print(f"Success! Serialized: {serialized}")
    print(f"Imports: {imports}")
except TypeError as e:
    print(f"FAILED with TypeError: {e}")
except Exception as e:
    print(f"FAILED with unexpected error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)

# Test case 2: Dictionary with only integers
test_dict_int = {1: 'value1', 2: 'value2'}
print(f"Test dict (int keys only): {test_dict_int}")
try:
    serialized, imports = serializer_factory(test_dict_int).serialize()
    print(f"Success! Serialized: {serialized}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print("\n" + "=" * 60)

# Test case 3: Dictionary with only strings
test_dict_str = {'key1': 'value1', 'key2': 'value2'}
print(f"Test dict (string keys only): {test_dict_str}")
try:
    serialized, imports = serializer_factory(test_dict_str).serialize()
    print(f"Success! Serialized: {serialized}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print("\n" + "=" * 60)

# Test case 4: Property-based test example
test_dict_prop = {1: 10, 'a': 20}
print(f"Test dict (from property test): {test_dict_prop}")
try:
    serialized, imports = serializer_factory(test_dict_prop).serialize()
    print(f"Success! Serialized: {serialized}")
    print(f"Imports: {imports}")
except TypeError as e:
    print(f"FAILED with TypeError: {e}")
except Exception as e:
    print(f"FAILED with unexpected error: {type(e).__name__}: {e}")