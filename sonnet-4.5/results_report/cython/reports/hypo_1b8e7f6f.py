from distutils.dist import Distribution
from unittest.mock import Mock
from hypothesis import given, assume, strategies as st
from Cython.Distutils import build_ext


@given(
    st.one_of(st.integers(), st.booleans(), st.lists(st.text()), st.dictionaries(st.text(), st.integers())),
    st.one_of(st.integers(), st.booleans(), st.lists(st.text()), st.dictionaries(st.text(), st.integers())),
)
def test_get_extension_attr_builder_takes_precedence(builder_value, ext_value):
    assume(builder_value != ext_value)
    assume(not builder_value and ext_value)

    dist = Distribution()
    builder = build_ext(dist)
    builder.cython_cplus = builder_value

    ext = Mock()
    ext.cython_cplus = ext_value

    result = builder.get_extension_attr(ext, 'cython_cplus')

    assert result == builder_value, f"Expected {builder_value}, got {result}"


if __name__ == "__main__":
    test_get_extension_attr_builder_takes_precedence()