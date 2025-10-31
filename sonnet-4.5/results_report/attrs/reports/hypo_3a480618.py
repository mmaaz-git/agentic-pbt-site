from hypothesis import given, strategies as st, assume
import attr
import inspect

@given(st.one_of(
    st.integers(),
    st.floats(allow_nan=False),
    st.text(),
    st.none(),
    st.lists(st.integers()),
    st.dictionaries(st.text(), st.integers()),
))
def test_has_should_raise_for_non_classes(non_class_value):
    """attr.has() should raise TypeError for non-class inputs per its documentation."""
    assume(not inspect.isclass(non_class_value))

    try:
        result = attr.has(non_class_value)
        raise AssertionError(
            f"attr.has({non_class_value!r}) returned {result} instead of raising TypeError"
        )
    except TypeError:
        pass

if __name__ == "__main__":
    test_has_should_raise_for_non_classes()