from hypothesis import given, strategies as st
import pandas.errors as errors

@given(st.sampled_from(["method", "classmethod", "staticmethod", "property"]))
def test_abstractmethoderror_valid_methodtype_works(methodtype):
    class DummyClass:
        pass

    instance = DummyClass()
    error = errors.AbstractMethodError(instance, methodtype=methodtype)
    error_str = str(error)

    assert methodtype in error_str
    assert "must be defined in the concrete class" in error_str

# Run the test
if __name__ == "__main__":
    test_abstractmethoderror_valid_methodtype_works()