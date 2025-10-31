from hypothesis import given
import hypothesis.strategies as st
import pandas.errors as pd_errors


@given(st.sampled_from(["method", "classmethod", "staticmethod", "property"]))
def test_abstract_method_error_valid_methodtype(valid_methodtype):
    class DummyClass:
        pass

    instance = DummyClass()
    error = pd_errors.AbstractMethodError(instance, methodtype=valid_methodtype)

    assert error.methodtype == valid_methodtype
    assert error.class_instance is instance

    error_str = str(error)
    assert isinstance(error_str, str)
    assert valid_methodtype in error_str
    assert "DummyClass" in error_str


if __name__ == "__main__":
    test_abstract_method_error_valid_methodtype()