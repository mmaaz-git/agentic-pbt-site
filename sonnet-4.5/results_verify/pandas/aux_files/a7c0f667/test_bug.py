from hypothesis import given, strategies as st, settings
from pandas.compat._optional import get_version
import types


@given(st.one_of(st.integers(), st.floats(), st.lists(st.integers()), st.none()))
@settings(max_examples=100)
def test_get_version_with_non_string_version(version_value):
    mock_module = types.ModuleType("test_module")
    mock_module.__version__ = version_value

    result = get_version(mock_module)
    assert isinstance(result, str)

# Run the test
test_get_version_with_non_string_version()