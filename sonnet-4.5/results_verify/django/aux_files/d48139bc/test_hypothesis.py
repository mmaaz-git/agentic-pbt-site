#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

import django
from django.conf import settings as django_settings

# Configure Django if not already configured
if not django_settings.configured:
    django_settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

from hypothesis import given, strategies as st, settings
from django.conf.urls.static import static
from django.core.exceptions import ImproperlyConfigured
import pytest

print("Running hypothesis test from bug report...")
print("This test expects all whitespace-only strings to raise ImproperlyConfigured")
print("="*60)

# The test from the bug report
@given(st.text(alphabet=' \t\n', min_size=1, max_size=10))
@settings(max_examples=50)
def test_static_whitespace_prefix_should_raise(prefix):
    """Test expects whitespace-only prefixes to raise ImproperlyConfigured"""
    with pytest.raises(ImproperlyConfigured):
        static(prefix)

# Run the test
try:
    test_static_whitespace_prefix_should_raise()
    print("Test passed! All whitespace-only strings raised ImproperlyConfigured")
except Exception as e:
    print(f"Test FAILED: {e}")

# Now test if the function actually accepts whitespace-only strings
print("\n" + "="*60)
print("Testing actual behavior with whitespace-only strings...")

test_cases = [' ', '\t', '\n', '  ', '\t\t', '\n\n', ' \t\n ']
for test_input in test_cases:
    repr_input = repr(test_input)
    try:
        result = static(test_input)
        print(f"static({repr_input}) returned: {result}")
        if result:
            print(f"  -> Created URL pattern: {result[0].pattern if hasattr(result[0], 'pattern') else 'N/A'}")
    except ImproperlyConfigured as e:
        print(f"static({repr_input}) raised: {e}")