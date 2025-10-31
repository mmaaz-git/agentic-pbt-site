#!/usr/bin/env python3
"""Minimal reproduction of django.db.backends.utils.truncate_name bug"""

import sys
import os
# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.backends.utils import truncate_name

# Test case that should fail
identifier = '00'
length = 1

result = truncate_name(identifier, length)

print(f"truncate_name({identifier!r}, {length}) = {result!r}")
print(f"Result length: {len(result)}")
print(f"Expected max length: {length}")
print(f"Violates contract: {len(result) > length}")

# Test idempotence issue
result2 = truncate_name(result, length)
print(f"\nIdempotence test:")
print(f"First call:  truncate_name({identifier!r}, {length}) = {result!r}")
print(f"Second call: truncate_name({result!r}, {length}) = {result2!r}")
print(f"Is idempotent: {result == result2}")

# Show the assertion failure
try:
    assert len(result) <= length, f"Result length {len(result)} exceeds requested length {length}"
except AssertionError as e:
    print(f"\nAssertion Error: {e}")