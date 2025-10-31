import sys
import pytest
from hypothesis import given, strategies as st, settings
from pandas.compat._optional import import_optional_dependency


@given(st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), min_codepoint=ord('a')), min_size=5))
@settings(max_examples=10)  # Reduced for testing
def test_import_optional_dependency_submodule_no_keyerror(parent_name):
    parent_module_name = f"fake_parent_{parent_name}"
    submodule_name = f"{parent_module_name}.submodule"

    class FakeSubmodule:
        __name__ = submodule_name
        __version__ = "1.0.0"

    sys.modules[submodule_name] = FakeSubmodule()

    if parent_module_name in sys.modules:
        del sys.modules[parent_module_name]

    try:
        with pytest.raises((ImportError, KeyError)) as exc_info:
            import_optional_dependency(submodule_name, errors="raise", min_version="0.0.1")

        if isinstance(exc_info.value, KeyError):
            raise AssertionError(f"KeyError raised instead of ImportError: {exc_info.value}")
    finally:
        if submodule_name in sys.modules:
            del sys.modules[submodule_name]

# Run with hypothesis test runner
if __name__ == "__main__":
    print("Running Hypothesis test...")
    test_import_optional_dependency_submodule_no_keyerror()
    print("Test completed")