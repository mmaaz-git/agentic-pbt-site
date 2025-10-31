#!/usr/bin/env python3
"""Minimal reproduction of Django AppConfig.create() IndexError bug."""

import os
import sys
import django
from django.conf import settings

# Configure minimal Django settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.auth',
        ],
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': ':memory:',
            }
        }
    )

# Setup Django
django.setup()

# Now reproduce the bug
from django.apps.config import AppConfig

try:
    # This should raise an IndexError instead of a proper error message
    result = AppConfig.create("django.contrib.auth.")
    print(f"Unexpectedly succeeded: {result}")
except IndexError as e:
    print(f"IndexError: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Other exception ({type(e).__name__}): {e}")
    import traceback
    traceback.print_exc()