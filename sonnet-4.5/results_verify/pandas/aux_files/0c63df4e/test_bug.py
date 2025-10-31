#!/usr/bin/env python3
"""Test script to reproduce the ExtensionDtype.construct_from_string bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.api.extensions import ExtensionDtype
from pandas.core.arrays import ExtensionArray


print("=" * 60)
print("Test 1: Creating ExtensionDtype with name as property")
print("=" * 60)

class PropertyNameDtype(ExtensionDtype):
    type = str

    @property
    def name(self):
        return "my_custom_dtype"

    @classmethod
    def construct_array_type(cls):
        return ExtensionArray


# Create an instance and verify name works
dtype = PropertyNameDtype()
print(f"Created dtype with name: {dtype.name}")
print(f"Type of dtype.name: {type(dtype.name)}")

# Check what cls.name returns
print(f"\nAccessing via class: PropertyNameDtype.name = {PropertyNameDtype.name}")
print(f"Type of PropertyNameDtype.name: {type(PropertyNameDtype.name)}")
print(f"Is PropertyNameDtype.name a string? {isinstance(PropertyNameDtype.name, str)}")

# Try to use construct_from_string
print("\n" + "=" * 60)
print("Test 2: Calling construct_from_string")
print("=" * 60)
try:
    result = PropertyNameDtype.construct_from_string("my_custom_dtype")
    print(f"Success: {result}")
except AssertionError as e:
    print(f"AssertionError: {e}")
    print(f"Expected: name to work as property, got assertion failure")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Test 3: ExtensionDtype with name as class attribute")
print("=" * 60)

class ClassAttrNameDtype(ExtensionDtype):
    type = str
    name = "class_attr_dtype"  # Name as class attribute

    @classmethod
    def construct_array_type(cls):
        return ExtensionArray


# Create an instance and verify name works
dtype2 = ClassAttrNameDtype()
print(f"Created dtype with name: {dtype2.name}")
print(f"Type of dtype2.name: {type(dtype2.name)}")

# Check what cls.name returns
print(f"\nAccessing via class: ClassAttrNameDtype.name = {ClassAttrNameDtype.name}")
print(f"Type of ClassAttrNameDtype.name: {type(ClassAttrNameDtype.name)}")
print(f"Is ClassAttrNameDtype.name a string? {isinstance(ClassAttrNameDtype.name, str)}")

# Try to use construct_from_string
print("\n" + "=" * 60)
print("Test 4: Calling construct_from_string with class attribute")
print("=" * 60)
try:
    result = ClassAttrNameDtype.construct_from_string("class_attr_dtype")
    print(f"Success: {result}")
    print(f"Result type: {type(result)}")
except AssertionError as e:
    print(f"AssertionError: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Test 5: Property-based test case from bug report")
print("=" * 60)

class PropertyNameDtype2(ExtensionDtype):
    type = str
    _test_name = "property_name_dtype"

    @property
    def name(self):
        return self._test_name

    @classmethod
    def construct_array_type(cls):
        return ExtensionArray


dtype3 = PropertyNameDtype2()
print(f"Created dtype with name: {dtype3.name}")

try:
    result = dtype3.construct_from_string("property_name_dtype")
    print(f"Success (using instance method): {result}")
except AssertionError as e:
    print(f"AssertionError (using instance method): {e}")
except Exception as e:
    print(f"Error (using instance method): {type(e).__name__}: {e}")

try:
    result = PropertyNameDtype2.construct_from_string("property_name_dtype")
    print(f"Success (using class method): {result}")
except AssertionError as e:
    print(f"AssertionError (using class method): {e}")
except Exception as e:
    print(f"Error (using class method): {type(e).__name__}: {e}")