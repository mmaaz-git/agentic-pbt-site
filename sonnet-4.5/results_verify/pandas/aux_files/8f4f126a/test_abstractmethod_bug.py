from hypothesis import given, strategies as st
import pandas as pd


class DummyClass:
    pass


@given(st.sampled_from(["method", "classmethod", "staticmethod", "property"]))
def test_abstractmethoderror_valid_methodtypes_should_not_crash(methodtype):
    instance = DummyClass()
    err = pd.errors.AbstractMethodError(instance, methodtype=methodtype)
    error_message = str(err)
    assert isinstance(error_message, str)
    assert len(error_message) > 0

if __name__ == "__main__":
    test_abstractmethoderror_valid_methodtypes_should_not_crash()