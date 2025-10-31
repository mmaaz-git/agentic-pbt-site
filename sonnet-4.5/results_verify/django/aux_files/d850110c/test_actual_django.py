#!/usr/bin/env python3
"""Test the actual Django Oracle backend for the typo"""

import os
import sys
import django
from django.conf import settings

# Configure Django settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.oracle',
                'NAME': 'test',
            }
        },
        SECRET_KEY='test-secret-key',
        INSTALLED_APPS=[],
    )
    django.setup()

from django.db.backends.oracle.operations import DatabaseOperations

print("Testing actual Django Oracle backend...")

# Create an instance
ops = DatabaseOperations(None)

# Test with invalid lookup type to trigger the error
try:
    ops.date_extract_sql("invalid!type", "field", ())
except ValueError as e:
    print(f"Error message from actual Django code: {e}")
    if "loookup" in str(e):
        print("✓ Bug confirmed in actual Django code: Error message contains 'loookup' with 3 o's")
    else:
        print("✗ Bug not found in actual Django: Error message does not contain the typo")
except Exception as e:
    print(f"Different exception raised: {type(e).__name__}: {e}")