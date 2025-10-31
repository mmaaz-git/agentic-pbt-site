#!/usr/bin/env python3
"""Test what happens when None is passed to various functions"""

import django
from django.conf import settings

# Configure Django
settings.configure(
    DEBUG=True,
    DATABASES={
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': ':memory:',
        }
    },
    INSTALLED_APPS=['django.contrib.contenttypes', 'django.contrib.auth'],
    SECRET_KEY='test-key',
)
django.setup()

from django.db.models.functions.text import Left, Right, LPad, RPad, Repeat, Substr
from django.db.models import Value
from django.db import models

# Create a test model to work with actual database queries
from django.contrib.auth.models import User

# Test what happens with expressions that resolve to None
print("Testing LPad with None length...")
try:
    # LPad handles None gracefully due to the explicit check
    lpad = LPad(Value('test'), None)
    print(f"LPad created successfully: {lpad}")
except Exception as e:
    print(f"Exception: {e}")

print("\nTesting RPad with None length...")
try:
    # RPad inherits from LPad so should have same behavior
    rpad = RPad(Value('test'), None)
    print(f"RPad created successfully: {rpad}")
except Exception as e:
    print(f"Exception: {e}")

print("\nTesting Repeat with None number...")
try:
    # Repeat handles None gracefully due to the explicit check
    repeat = Repeat(Value('test'), None)
    print(f"Repeat created successfully: {repeat}")
except Exception as e:
    print(f"Exception: {e}")

print("\nTesting Substr with None pos...")
try:
    # Substr has the same bug as Left
    substr = Substr(Value('test'), None)
    print(f"Substr created successfully: {substr}")
except TypeError as e:
    print(f"TypeError: {e}")
except Exception as e:
    print(f"Exception: {e}")

print("\nTesting Substr with valid pos but None length...")
try:
    # This should work as length is optional
    substr = Substr(Value('test'), 1, None)
    print(f"Substr created successfully: {substr}")
except Exception as e:
    print(f"Exception: {e}")