#!/usr/bin/env python3
"""Test script to reproduce the AbstractMethodError bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pandas as pd

print("Testing pandas.errors.AbstractMethodError bug...")
print()

class DummyClass:
    pass

instance = DummyClass()

print("Attempting to create AbstractMethodError with invalid methodtype='invalid_type'...")
try:
    pd.errors.AbstractMethodError(instance, methodtype="invalid_type")
except ValueError as e:
    print(f"Actual error message: {e}")
    print()
    print("Expected error message: methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got invalid_type instead.")
    print()

    # Check if the error message has swapped parameters
    error_str = str(e)
    if "methodtype must be one of invalid_type" in error_str:
        print("BUG CONFIRMED: The error message has swapped parameters!")
        print("The format string incorrectly shows the invalid value where it should show valid options.")
    else:
        print("Bug not confirmed - error message appears correct.")

print("\n" + "="*60)
print("Testing with the hypothesis test...")
print("="*60)
print()

from hypothesis import given, strategies as st
import pytest

@given(st.text().filter(lambda x: x not in {"method", "classmethod", "staticmethod", "property"}))
def test_abstract_method_error_validation_message(invalid_methodtype):
    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        pd.errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    valid_types = {"method", "classmethod", "staticmethod", "property"}

    # The bug: error message has swapped parameters
    if f"methodtype must be one of {invalid_methodtype}" in error_message:
        print(f"Bug confirmed with input '{invalid_methodtype}': {error_message}")
        return False

    return True

# Run a few test cases
test_inputs = ["invalid_type", "foo", "bar", "unknown", "xyz123", ""]
for test_input in test_inputs:
    if test_input not in {"method", "classmethod", "staticmethod", "property"}:
        try:
            result = test_abstract_method_error_validation_message(test_input)
            if not result:
                print(f"Test failed for input: '{test_input}'")
        except:
            pass