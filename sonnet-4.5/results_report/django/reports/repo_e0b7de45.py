#!/usr/bin/env python3
"""
Minimal reproduction of django.db.backends.sqlite3._functions._sqlite_lpad bug
Shows that the function violates its length invariant when fill_text is empty
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
from django.db.backends.sqlite3._functions import _sqlite_lpad

print("Testing _sqlite_lpad function with empty fill_text")
print("=" * 60)

# Test case 1: Empty text, length 1, empty fill_text
print("\nTest 1: _sqlite_lpad('', 1, '')")
result1 = _sqlite_lpad('', 1, '')
print(f"  Result: '{result1}'")
print(f"  Length: {len(result1)}")
print(f"  Expected length: 1")
print(f"  FAIL: Expected length 1, got {len(result1)}")

# Test case 2: 'hello', length 10, empty fill_text
print("\nTest 2: _sqlite_lpad('hello', 10, '')")
result2 = _sqlite_lpad('hello', 10, '')
print(f"  Result: '{result2}'")
print(f"  Length: {len(result2)}")
print(f"  Expected length: 10")
print(f"  FAIL: Expected length 10, got {len(result2)}")

# Test case 3: 'x', length 5, empty fill_text
print("\nTest 3: _sqlite_lpad('x', 5, '')")
result3 = _sqlite_lpad('x', 5, '')
print(f"  Result: '{result3}'")
print(f"  Length: {len(result3)}")
print(f"  Expected length: 5")
print(f"  FAIL: Expected length 5, got {len(result3)}")

# Control test: Non-empty fill_text works correctly
print("\nControl Test: _sqlite_lpad('hello', 10, '*')")
result4 = _sqlite_lpad('hello', 10, '*')
print(f"  Result: '{result4}'")
print(f"  Length: {len(result4)}")
print(f"  Expected length: 10")
print(f"  PASS: Got expected length 10")

print("\n" + "=" * 60)
print("Summary: The function violates its fundamental invariant")
print("LPAD should always return a string of exactly 'length' characters")
print("but returns the unpadded text when fill_text is empty.")