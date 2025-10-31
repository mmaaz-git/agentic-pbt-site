#!/usr/bin/env python3
"""Test to reproduce the _sqlite_sqrt bug with negative inputs."""

import sys
import os

# Add Django to the path
django_path = "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages"
sys.path.insert(0, django_path)

from django.db.backends.sqlite3._functions import _sqlite_sqrt

# Test 1: Simple reproduction
print("Test 1: Attempting _sqlite_sqrt(-1)...")
try:
    result = _sqlite_sqrt(-1)
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError caught: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Test 2: Test with None (should return None)
print("\nTest 2: _sqlite_sqrt(None)...")
result = _sqlite_sqrt(None)
print(f"Result: {result}")

# Test 3: Test with positive number (should work)
print("\nTest 3: _sqlite_sqrt(4)...")
result = _sqlite_sqrt(4)
print(f"Result: {result}")

# Test 4: Test with zero (should work)
print("\nTest 4: _sqlite_sqrt(0)...")
result = _sqlite_sqrt(0)
print(f"Result: {result}")

# Test 5: Test with various negative numbers
print("\nTest 5: Testing various negative numbers...")
test_values = [-1.0, -10, -0.5, -100]
for val in test_values:
    try:
        result = _sqlite_sqrt(val)
        print(f"_sqlite_sqrt({val}) = {result}")
    except ValueError as e:
        print(f"_sqlite_sqrt({val}) raised ValueError: {e}")