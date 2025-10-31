#!/usr/bin/env python3
"""Test with existing modules to trigger the ViewDoesNotExist exception"""

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

# Test cases with modules that exist but attributes that don't
test_cases = [
    ('os.path', 'nonexistent_view'),  # os.path module exists, but nonexistent_view doesn't
    ('sys', 'fake_function'),          # sys module exists, but fake_function doesn't
    ('os', 'not_a_real_func'),         # os module exists, but not_a_real_func doesn't
    ('django', 'foobar'),               # django module exists, but foobar doesn't
]

print("Testing with existing modules but non-existent attributes:")
print("=" * 70)

for module_name, view_name in test_cases:
    csrf_view = f"{module_name}.{view_name}"
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
        print(f"  ImportError raised (expected): {e}")
    except Exception as e:
        print(f"  Other exception: {type(e).__name__}: {e}")