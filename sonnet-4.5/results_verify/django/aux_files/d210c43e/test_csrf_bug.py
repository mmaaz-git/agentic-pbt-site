#!/usr/bin/env python3
"""Test the reported CSRF check bug"""

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
CSRF_FAILURE_VIEW = 'os.path.nonexistent_view'
INSTALLED_APPS = []
USE_TZ = False
""")

import django
django.setup()

from django.core.checks.security.csrf import check_csrf_failure_view
from django.core.exceptions import ViewDoesNotExist

print("Testing check_csrf_failure_view with CSRF_FAILURE_VIEW = 'os.path.nonexistent_view'")
print("-" * 70)

try:
    result = check_csrf_failure_view(None)
    print(f"SUCCESS: Function returned normally")
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    if result:
        for error in result:
            print(f"  Error ID: {error.id}")
            print(f"  Error msg: {error.msg}")
except ViewDoesNotExist as e:
    print(f"CRASH: ViewDoesNotExist exception raised!")
    print(f"Exception: {e}")
except ImportError as e:
    print(f"ImportError raised: {e}")
except Exception as e:
    print(f"Other exception raised: {type(e).__name__}: {e}")