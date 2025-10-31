#!/usr/bin/env python3
"""Reproduce the reported bugs in pandas.errors.AbstractMethodError"""
import pandas as pd
import sys

print("Testing pandas.errors.AbstractMethodError bugs")
print("=" * 60)

# Bug 1: Test swapped error message parameters
print("\nBug 1: Testing swapped error message parameters")
print("-" * 40)

class DummyClass:
    pass

instance = DummyClass()

try:
    error = pd.errors.AbstractMethodError(instance, methodtype='invalid_type')
except ValueError as e:
    print(f"Error message: {e}")
    print(f"Expected: methodtype must be one of {{'staticmethod', 'method', 'classmethod', 'property'}}, got invalid_type instead.")
    print(f"Actual:   {e}")

# Bug 2: Test AttributeError in __str__ for classmethod
print("\nBug 2: Testing AttributeError in __str__ for classmethod")
print("-" * 40)

try:
    error = pd.errors.AbstractMethodError(instance, methodtype='classmethod')
    error_str = str(error)
    print(f"Error string successfully generated: {error_str}")
except AttributeError as e:
    print(f"AttributeError occurred: {e}")
    print("This happens because instance doesn't have __name__ attribute")

# Additional test: Verify proper behavior with actual class
print("\nAdditional test: Testing with actual class for classmethod")
print("-" * 40)

try:
    error = pd.errors.AbstractMethodError(DummyClass, methodtype='classmethod')
    error_str = str(error)
    print(f"With class object: {error_str}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Test hypothesis test case
print("\nRunning property-based test")
print("-" * 40)

def test_abstract_method_error_message_clarity(invalid_methodtype):
    class DummyClass:
        pass

    instance = DummyClass()

    try:
        pd.errors.AbstractMethodError(instance, methodtype=invalid_methodtype)
    except ValueError as e:
        error_message = str(e)
        print(f"Error message with '{invalid_methodtype}': {error_message}")
        # This assertion checks if the invalid methodtype appears in the error message
        if invalid_methodtype in error_message:
            print(f"Testing with invalid_methodtype='{invalid_methodtype}': PASS - found in message")
            return True
        else:
            print(f"Testing with invalid_methodtype='{invalid_methodtype}': FAIL - not found in message")
            return False

# Run with specific failing case 'x'
result = test_abstract_method_error_message_clarity('x')
if not result:
    print("Bug confirmed: invalid methodtype 'x' not found in error message")