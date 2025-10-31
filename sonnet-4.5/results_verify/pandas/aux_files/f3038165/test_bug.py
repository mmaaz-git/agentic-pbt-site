#!/usr/bin/env python3
import pandas.errors


class DummyClass:
    pass


instance = DummyClass()

print("Testing AbstractMethodError with invalid methodtype='invalid_type'...")
try:
    pandas.errors.AbstractMethodError(instance, methodtype='invalid_type')
except ValueError as e:
    print(f"Error message received: {str(e)}")
    print()

    # Check what we expect vs what we get
    print("Expected message format:")
    print("methodtype must be one of {'staticmethod', 'classmethod', 'property', 'method'}, got invalid_type instead.")
    print()
    print("Actual message received:")
    print(str(e))
    print()

    # Check if the message parts are in the right order
    if "methodtype must be one of invalid_type" in str(e):
        print("BUG CONFIRMED: The error message has swapped the variables!")
        print("The invalid value 'invalid_type' is shown where the valid options should be.")
    else:
        print("Bug not reproduced - message format seems correct.")