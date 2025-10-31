from hypothesis import given, strategies as st
from Cython.Utility.Dataclasses import field, MISSING


@given(st.booleans())
def test_field_repr_uses_correct_attribute_name(kw_only_value):
    f = field(default=MISSING, kw_only=kw_only_value)
    repr_str = repr(f)

    assert f'kw_only={kw_only_value!r}' in repr_str, \
        f"Expected 'kw_only={kw_only_value!r}' in repr, but got: {repr_str}"

# Run the test
test_field_repr_uses_correct_attribute_name()