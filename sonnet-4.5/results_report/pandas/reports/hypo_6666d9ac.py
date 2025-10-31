from hypothesis import given, strategies as st
from pandas.compat._optional import import_optional_dependency
import pytest


builtin_modules = st.sampled_from(['sys', 'os', 'io', 'math', 'json', 're', 'time'])


@given(builtin_modules, st.text(min_size=1, max_size=10))
def test_errors_ignore_never_raises_on_version_check(module_name, min_version):
    result = import_optional_dependency(module_name, min_version=min_version, errors='ignore')


@given(builtin_modules, st.text(min_size=1, max_size=10))
def test_errors_warn_never_raises_on_version_check(module_name, min_version):
    with pytest.warns():
        result = import_optional_dependency(module_name, min_version=min_version, errors='warn')

if __name__ == "__main__":
    # Run the tests
    test_errors_ignore_never_raises_on_version_check()
    test_errors_warn_never_raises_on_version_check()