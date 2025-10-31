from hypothesis import given, strategies as st
import pandas.errors as pd_errors


class DummyClass:
    pass


@given(st.sampled_from(["method", "classmethod", "staticmethod", "property"]))
def test_abstract_method_error_valid_methodtype(valid_methodtype):
    dummy = DummyClass()
    error = pd_errors.AbstractMethodError(dummy, methodtype=valid_methodtype)
    assert error.methodtype == valid_methodtype
    assert error.class_instance is dummy

    error_message = str(error)
    assert valid_methodtype in error_message
    assert "DummyClass" in error_message

# Run the test
test_abstract_method_error_valid_methodtype()