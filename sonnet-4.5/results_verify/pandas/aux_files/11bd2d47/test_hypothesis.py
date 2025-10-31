from hypothesis import given, strategies as st
import pandas.errors


@given(st.sampled_from(['method', 'classmethod', 'staticmethod', 'property']))
def test_abstract_method_error_valid_types(methodtype):
    class DummyClass:
        pass

    instance = DummyClass()
    error = pandas.errors.AbstractMethodError(instance, methodtype=methodtype)

    error_str = str(error)
    assert methodtype in error_str
    assert 'DummyClass' in error_str


if __name__ == "__main__":
    # Run the property-based test
    test_abstract_method_error_valid_types()