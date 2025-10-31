#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, settings, strategies as st
from pandas.compat._optional import import_optional_dependency

@given(min_version=st.text(min_size=1).filter(lambda s: s[0].isdigit()))
@settings(max_examples=10)  # Reduced for faster testing
def test_errors_ignore_returns_module_even_if_old(min_version):
    result = import_optional_dependency("numpy", min_version="999.0.0", errors="ignore")
    assert result is not None, (
        f"errors='ignore' should return module even if version is too old, "
        f"but got {result} for min_version={min_version}"
    )

# Run the test
try:
    test_errors_ignore_returns_module_even_if_old()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Test error: {e}")