#!/usr/bin/env python3
"""
Minimal reproduction case for Django Left/Right TypeError with None length
"""

import django
from django.conf import settings

# Configure Django settings
settings.configure(
    DEBUG=True,
    SECRET_KEY='test-key'
)
django.setup()

from django.db.models import Value
from django.db.models.functions.text import Left, Right

print("Testing Left with None length:")
print("-" * 40)
try:
    left_func = Left(Value('test'), None)
    print(f"Success: Created Left function with None length: {left_func}")
except TypeError as e:
    print(f"TypeError raised: {e}")
except ValueError as e:
    print(f"ValueError raised: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

print("\nTesting Right with None length:")
print("-" * 40)
try:
    right_func = Right(Value('test'), None)
    print(f"Success: Created Right function with None length: {right_func}")
except TypeError as e:
    print(f"TypeError raised: {e}")
except ValueError as e:
    print(f"ValueError raised: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

print("\nFor comparison - Testing Left with 0 length (should raise ValueError):")
print("-" * 40)
try:
    left_func = Left(Value('test'), 0)
    print(f"Success: Created Left function with 0 length: {left_func}")
except ValueError as e:
    print(f"ValueError raised (as expected): {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
