import pandas.errors
from hypothesis import given, strategies as st
import pytest


class SampleClass:
    pass


@given(st.sampled_from(["classmethod"]))
def test_abstractmethoderror_str_crashes_with_instance_for_classmethod(methodtype):
    instance = SampleClass()
    error = pandas.errors.AbstractMethodError(instance, methodtype=methodtype)

    with pytest.raises(AttributeError, match="'SampleClass' object has no attribute '__name__'"):
        str(error)

# Run the test
test_abstractmethoderror_str_crashes_with_instance_for_classmethod()