from hypothesis import given, strategies as st
import pandas.errors


class DummyClass:
    pass


@given(st.sampled_from(["method", "classmethod", "staticmethod", "property"]))
def test_abstract_method_error_valid_methodtype(valid_type):
    err = pandas.errors.AbstractMethodError(DummyClass(), methodtype=valid_type)
    assert err.methodtype == valid_type

    msg = str(err)
    assert valid_type in msg
    assert "DummyClass" in msg
    assert "must be defined in the concrete class" in msg


if __name__ == "__main__":
    test_abstract_method_error_valid_methodtype()