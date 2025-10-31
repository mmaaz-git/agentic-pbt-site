#!/usr/bin/env python3
"""Test with non-existent modules to see ImportError handling"""

import sys
import os

# Setup Django environment
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
os.environ['DJANGO_SETTINGS_MODULE'] = 'test_settings'

# Create minimal settings
with open('test_settings.py', 'w') as f:
    f.write("""
SECRET_KEY = 'test-key-12345678901234567890123456789012345678901234567890'
MIDDLEWARE = ['django.middleware.csrf.CsrfViewMiddleware']
CSRF_FAILURE_VIEW = 'django.views.defaults.csrf_failure'
INSTALLED_APPS = []
USE_TZ = False
""")

import django
django.setup()

from django.conf import settings
from django.core.checks.security.csrf import check_csrf_failure_view
from django.core.exceptions import ViewDoesNotExist

# Test cases with non-existent modules
test_cases = [
    'nonexistent_module.some_view',
    'fake_package.fake_view',
    'not_a_real_module.anything'
]

print("Testing with non-existent modules (should catch ImportError):")
print("=" * 70)

for csrf_view in test_cases:
    settings.CSRF_FAILURE_VIEW = csrf_view
    print(f"\nTesting CSRF_FAILURE_VIEW = '{csrf_view}'")
    print("-" * 50)

    try:
        result = check_csrf_failure_view(None)
        print(f"✓ Returned normally: {type(result)}")
        if result:
            for error in result:
                print(f"  - Error ID: {error.id}")
                print(f"    Message: {error.msg}")
    except ViewDoesNotExist as e:
        print(f"✗ BUG: ViewDoesNotExist exception raised!")
        print(f"  Exception message: {e}")
    except ImportError as e:
        print(f"✗ Unhandled ImportError (this is also a bug!): {e}")
    except Exception as e:
        print(f"  Other exception: {type(e).__name__}: {e}")