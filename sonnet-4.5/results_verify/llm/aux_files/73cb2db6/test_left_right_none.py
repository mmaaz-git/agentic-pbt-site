#!/usr/bin/env python3
"""Test the Left/Right functions with None length parameter"""

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
    INSTALLED_APPS=[],
)
django.setup()

from django.db.models.functions.text import Left, Right
from django.db.models import Value

# Test 1: Left with None length
print("Testing Left with None length...")
try:
    left = Left(Value('test'), None)
    print("No exception raised! Bug not reproduced.")
except TypeError as e:
    print(f"TypeError raised: {e}")
except ValueError as e:
    print(f"ValueError raised: {e}")
except Exception as e:
    print(f"Unexpected exception: {e}")

print()

# Test 2: Right with None length
print("Testing Right with None length...")
try:
    right = Right(Value('test'), None)
    print("No exception raised! Bug not reproduced.")
except TypeError as e:
    print(f"TypeError raised: {e}")
except ValueError as e:
    print(f"ValueError raised: {e}")
except Exception as e:
    print(f"Unexpected exception: {e}")

print()

# Test 3: Left with 0 length (should raise ValueError)
print("Testing Left with length=0...")
try:
    left = Left(Value('test'), 0)
    print("No exception raised!")
except ValueError as e:
    print(f"ValueError raised as expected: {e}")
except Exception as e:
    print(f"Unexpected exception: {e}")

print()

# Test 4: Left with negative length (should raise ValueError)
print("Testing Left with length=-1...")
try:
    left = Left(Value('test'), -1)
    print("No exception raised!")
except ValueError as e:
    print(f"ValueError raised as expected: {e}")
except Exception as e:
    print(f"Unexpected exception: {e}")