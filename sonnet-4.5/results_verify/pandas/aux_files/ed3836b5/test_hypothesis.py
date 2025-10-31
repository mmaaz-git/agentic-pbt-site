from hypothesis import given, strategies as st, example
import pandas as pd
import pytest


class DummyClass:
    pass


# Test with specific examples
print("Testing specific invalid methodtype values...")
invalid_values = ["invalid", "foo", "bar", "test", "xyz"]

for invalid_value in invalid_values:
    instance = DummyClass()
    valid_types = {"method", "classmethod", "staticmethod", "property"}

    try:
        pd.errors.AbstractMethodError(instance, methodtype=invalid_value)
        print(f"ERROR: Should have raised ValueError for '{invalid_value}'")
    except ValueError as e:
        error_msg = str(e)
        print(f"Input: '{invalid_value}'")
        print(f"Error: {error_msg}")

        # Check if the error message is backwards
        if f"methodtype must be one of {invalid_value}" in error_msg:
            print("  -> BUG CONFIRMED: Error message has swapped variables!")
        else:
            print("  -> Error message is correct")
        print()