#!/usr/bin/env python3
"""Proper Hypothesis test for the bug"""

from hypothesis import given, strategies as st
import pandas.errors

@given(st.sampled_from(["method", "classmethod", "staticmethod", "property"]))
def test_abstract_method_error_str_works_for_all_methodtypes(methodtype):
    class TestClass:
        pass

    instance = TestClass()
    error = pandas.errors.AbstractMethodError(instance, methodtype=methodtype)

    error_str = str(error)
    assert isinstance(error_str, str)
    assert len(error_str) > 0

# Run the test
if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_abstract_method_error_str_works_for_all_methodtypes()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")