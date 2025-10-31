#!/usr/bin/env python3

import sys
import traceback

# Direct reproduction test
print("=" * 60)
print("Direct reproduction test:")
print("=" * 60)

try:
    from pandas.compat._optional import import_optional_dependency

    print("\nTest 1: Calling import_optional_dependency('sys', min_version='1.0.0', errors='ignore')")
    try:
        result = import_optional_dependency('sys', min_version='1.0.0', errors='ignore')
        print(f"Success! Result: {result}")
        print(f"Result type: {type(result)}")
    except Exception as e:
        print(f"FAILED with exception: {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\nTest 2: Calling import_optional_dependency('sys', min_version='1.0.0', errors='warn')")
    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = import_optional_dependency('sys', min_version='1.0.0', errors='warn')
            print(f"Success! Result: {result}")
            if w:
                print(f"Warnings captured: {[str(warning.message) for warning in w]}")
    except Exception as e:
        print(f"FAILED with exception: {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\nTest 3: Calling import_optional_dependency('sys', min_version='1.0.0', errors='raise')")
    try:
        result = import_optional_dependency('sys', min_version='1.0.0', errors='raise')
        print(f"Success! Result: {result}")
    except Exception as e:
        print(f"Expected exception with errors='raise': {type(e).__name__}: {e}")

except ImportError as e:
    print(f"Could not import pandas: {e}")

# Hypothesis test
print("\n" + "=" * 60)
print("Hypothesis property-based test:")
print("=" * 60)

try:
    from hypothesis import given, strategies as st
    from pandas.compat._optional import import_optional_dependency
    import pytest

    builtin_modules = st.sampled_from(['sys', 'os', 'io', 'math', 'json', 're', 'time'])

    @given(builtin_modules, st.text(min_size=1, max_size=10))
    def test_errors_ignore_never_raises_on_version_check(module_name, min_version):
        """Test that errors='ignore' never raises an exception"""
        try:
            result = import_optional_dependency(module_name, min_version=min_version, errors='ignore')
            print(f"✓ Test passed for module={module_name}, min_version={min_version[:20] if len(min_version) > 20 else min_version}")
            return True
        except Exception as e:
            print(f"✗ Test FAILED for module={module_name}, min_version={min_version[:20] if len(min_version) > 20 else min_version}")
            print(f"  Exception: {type(e).__name__}: {e}")
            raise

    @given(builtin_modules, st.text(min_size=1, max_size=10))
    def test_errors_warn_never_raises_on_version_check(module_name, min_version):
        """Test that errors='warn' never raises an exception"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = import_optional_dependency(module_name, min_version=min_version, errors='warn')
            print(f"✓ Test passed for module={module_name}, min_version={min_version[:20] if len(min_version) > 20 else min_version}")
            return True
        except Exception as e:
            print(f"✗ Test FAILED for module={module_name}, min_version={min_version[:20] if len(min_version) > 20 else min_version}")
            print(f"  Exception: {type(e).__name__}: {e}")
            raise

    # Run the hypothesis tests
    print("\nRunning hypothesis test for errors='ignore':")
    try:
        test_errors_ignore_never_raises_on_version_check()
    except Exception as e:
        print(f"Hypothesis test failed!")

    print("\nRunning hypothesis test for errors='warn':")
    try:
        test_errors_warn_never_raises_on_version_check()
    except Exception as e:
        print(f"Hypothesis test failed!")

except ImportError as e:
    print(f"Could not run hypothesis tests: {e}")

# Additional test to check sys module specifics
print("\n" + "=" * 60)
print("Checking sys module version attribute:")
print("=" * 60)

print(f"sys module has __version__: {hasattr(sys, '__version__')}")
print(f"sys.version: {sys.version}")
print(f"sys.version_info: {sys.version_info}")