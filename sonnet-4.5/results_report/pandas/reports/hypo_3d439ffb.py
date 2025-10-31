from hypothesis import given, strategies as st
from pandas.compat._optional import import_optional_dependency

def test_errors_ignore_returns_module_when_old_version():
    """Test that errors='ignore' returns the module even when version is too old.

    According to the docstring, when errors='ignore':
    'return the module, even if the version is too old.'
    """
    result = import_optional_dependency("hypothesis", errors="ignore", min_version="999.0.0")
    assert result is not None, "errors='ignore' should return module even when version is too old"
    print(f"Test 1 FAILED: Expected module, got {result}")

def test_errors_ignore_module_without_version():
    """Test that errors='ignore' handles modules without __version__ gracefully.

    According to the contract, errors='ignore' should not raise exceptions.
    """
    try:
        result = import_optional_dependency("sys", errors="ignore", min_version="1.0.0")
        assert result is not None, "errors='ignore' should not raise even if module has no __version__"
        print(f"Test 2 passed: Got result {result}")
    except ImportError as e:
        print(f"Test 2 FAILED: errors='ignore' raised ImportError: {e}")

# Run the tests
print("Running property-based tests for import_optional_dependency with errors='ignore':")
print("-" * 70)
test_errors_ignore_returns_module_when_old_version()
test_errors_ignore_module_without_version()