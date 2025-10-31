#!/usr/bin/env python3
"""Hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
from django.conf.urls import include
from django.core.exceptions import ImproperlyConfigured

@given(st.text(min_size=1))
@settings(max_examples=10)
def test_empty_string_app_name_should_be_valid_with_namespace(namespace):
    patterns = []

    try:
        result = include((patterns, ''), namespace=namespace)
        urlconf_module, app_name, ns = result

        assert app_name == '', f"Expected app_name to be '', got {app_name!r}"
        assert ns == namespace, f"Expected namespace to be {namespace!r}, got {ns!r}"
        print(f"✓ Test passed with namespace={namespace!r}")
    except ImproperlyConfigured as e:
        print(f"✗ Test FAILED with namespace={namespace!r}: {e}")
        raise AssertionError(f"include() incorrectly raised ImproperlyConfigured for empty string app_name with namespace={namespace!r}")

# Run the test
if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_empty_string_app_name_should_be_valid_with_namespace()
        print("\nAll tests passed!")
    except AssertionError as e:
        print(f"\nTest suite failed: {e}")