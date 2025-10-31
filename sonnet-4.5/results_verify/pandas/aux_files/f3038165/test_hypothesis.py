#!/usr/bin/env python3
from hypothesis import given, strategies as st
import pandas.errors
import pytest


@given(st.text().filter(lambda x: x not in ['method', 'classmethod', 'staticmethod', 'property']))
def test_abstract_method_error_invalid_types(invalid_methodtype):
    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        pandas.errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    # The bug report claims these assertions should pass, let's check:
    print(f"Testing with methodtype='{invalid_methodtype}'")
    print(f"Error message: {error_message}")

    assert 'methodtype must be one of' in error_message
    assert invalid_methodtype in error_message
    valid_types = {'method', 'classmethod', 'staticmethod', 'property'}
    for valid_type in valid_types:
        assert valid_type in error_message
    print("All assertions passed for this input")


# Run the test with a specific example
test_abstract_method_error_invalid_types('invalid_type')