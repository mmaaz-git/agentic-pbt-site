from hypothesis import given, strategies as st
from Cython.Build.Inline import safe_type

class CustomClass:
    pass

@given(st.builds(CustomClass))
def test_safe_type_with_custom_class_no_context(obj):
    result = safe_type(obj)
    assert isinstance(result, str)

# Run the test
test_safe_type_with_custom_class_no_context()