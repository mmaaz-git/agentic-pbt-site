#!/usr/bin/env python3
"""Test the reported CSRF check bug with Hypothesis"""

import sys
import os

# Setup Django environment
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Create minimal settings first
with open('test_settings2.py', 'w') as f:
    f.write("""
SECRET_KEY = 'test-key-12345678901234567890123456789012345678901234567890'
MIDDLEWARE = ['django.middleware.csrf.CsrfViewMiddleware']
CSRF_FAILURE_VIEW = 'django.views.defaults.csrf_failure'  # Default value
INSTALLED_APPS = []
USE_TZ = False
""")

os.environ['DJANGO_SETTINGS_MODULE'] = 'test_settings2'

import django
django.setup()

from hypothesis import given, strategies as st, assume, settings as hyp_settings
from django.conf import settings
from django.core.checks.security.csrf import check_csrf_failure_view
from django.core.exceptions import ViewDoesNotExist

@given(st.text(min_size=1), st.text(min_size=1))
@hyp_settings(max_examples=20, deadline=None)
def test_csrf_failure_view_should_not_crash(module_name, view_name):
    assume('.' not in module_name)
    assume('.' not in view_name)

    # Skip names that might cause import errors
    assume(not module_name.startswith('_'))
    assume(not view_name.startswith('_'))

    settings.CSRF_FAILURE_VIEW = f'{module_name}.{view_name}'

    try:
        result = check_csrf_failure_view(None)
        assert isinstance(result, list), "Should return a list of errors"
        print(f"✓ {settings.CSRF_FAILURE_VIEW}: Returned {len(result)} errors")
    except ViewDoesNotExist:
        print(f"✗ BUG FOUND: {settings.CSRF_FAILURE_VIEW} raised ViewDoesNotExist!")
        raise AssertionError(
            f"BUG: check_csrf_failure_view raised ViewDoesNotExist for "
            f"'{settings.CSRF_FAILURE_VIEW}' instead of returning error list"
        )
    except ImportError:
        # This is expected for non-existent modules
        print(f"  {settings.CSRF_FAILURE_VIEW}: ImportError (expected)")

# Test the specific failure case mentioned in the bug report
print("Testing specific failure case: __main__.0")
print("-" * 70)
settings.CSRF_FAILURE_VIEW = '__main__.0'
try:
    result = check_csrf_failure_view(None)
    print(f"Result: {result}")
except ViewDoesNotExist as e:
    print(f"CONFIRMED BUG: ViewDoesNotExist raised for '__main__.0'")
    print(f"Exception: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

print("\nRunning property-based tests...")
print("-" * 70)
try:
    test_csrf_failure_view_should_not_crash()
    print("\nAll tests passed!")
except AssertionError as e:
    print(f"\nTest failed with: {e}")