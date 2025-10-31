#!/usr/bin/env python3
"""Test script to reproduce the AbstractMethodError bug"""

import pandas.errors
import traceback
from hypothesis import given, strategies as st

# First, run the simple reproduction case
print("=== Simple Reproduction Case ===")
class MyClass:
    pass

error = pandas.errors.AbstractMethodError(MyClass(), methodtype="classmethod")

try:
    message = str(error)
    print(f"Success: {message}")
except AttributeError as e:
    print(f"AttributeError: {e}")
    traceback.print_exc()

print("\n=== Testing all methodtypes ===")
# Test each methodtype
for methodtype in ["method", "classmethod", "staticmethod", "property"]:
    print(f"\nTesting methodtype='{methodtype}':")
    class TestClass:
        pass

    instance = TestClass()
    error = pandas.errors.AbstractMethodError(instance, methodtype=methodtype)

    try:
        error_str = str(error)
        print(f"  Success: {error_str}")
    except AttributeError as e:
        print(f"  Failed with AttributeError: {e}")

print("\n=== Property-Based Test ===")
# Run the property-based test
@given(st.sampled_from(["method", "classmethod", "staticmethod", "property"]))
def test_abstract_method_error_str_works_for_all_methodtypes(methodtype):
    class TestClass:
        pass

    instance = TestClass()
    error = pandas.errors.AbstractMethodError(instance, methodtype=methodtype)

    error_str = str(error)
    assert isinstance(error_str, str)
    assert len(error_str) > 0
    return True

try:
    test_abstract_method_error_str_works_for_all_methodtypes()
    print("Property-based test passed!")
except Exception as e:
    print(f"Property-based test failed: {e}")
    traceback.print_exc()

print("\n=== Testing with actual class objects ===")
# Let's also test with actual class objects (not instances)
for methodtype in ["method", "classmethod", "staticmethod", "property"]:
    print(f"\nTesting methodtype='{methodtype}' with class object:")
    class TestClass:
        pass

    error = pandas.errors.AbstractMethodError(TestClass, methodtype=methodtype)

    try:
        error_str = str(error)
        print(f"  Success: {error_str}")
    except AttributeError as e:
        print(f"  Failed with AttributeError: {e}")