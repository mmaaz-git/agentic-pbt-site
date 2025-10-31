from hypothesis import given, strategies as st
from flask.helpers import get_root_path

@given(st.sampled_from(['sys', 'builtins', 'marshal']))
def test_get_root_path_should_not_raise_for_builtin_modules(module_name):
    result = get_root_path(module_name)
    assert isinstance(result, str)