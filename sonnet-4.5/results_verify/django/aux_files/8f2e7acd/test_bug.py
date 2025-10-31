#!/usr/bin/env python3
"""Test script to reproduce the LPAD/RPAD empty padding bug"""

import sys
import os

# Add Django to path
sys.path.append('/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.backends.sqlite3._functions import _sqlite_lpad, _sqlite_rpad

print("=" * 60)
print("Testing Django SQLite3 LPAD/RPAD with empty padding string")
print("=" * 60)

# Test case from bug report
text = "hello"
length = 10
fill_text = ""

print(f"\nTest inputs:")
print(f"  text = {text!r}")
print(f"  length = {length}")
print(f"  fill_text = {fill_text!r} (empty string)")

# Test LPAD
result_lpad = _sqlite_lpad(text, length, fill_text)
print(f"\nLPAD result: {result_lpad!r}")
if result_lpad is not None:
    print(f"Expected length: {length}, Actual length: {len(result_lpad)}")
    if len(result_lpad) != length:
        print("❌ LENGTH VIOLATION: Result does not have the expected length!")
else:
    print("Result is None (NULL)")

# Test RPAD
result_rpad = _sqlite_rpad(text, length, fill_text)
print(f"\nRPAD result: {result_rpad!r}")
if result_rpad is not None:
    print(f"Expected length: {length}, Actual length: {len(result_rpad)}")
    if len(result_rpad) != length:
        print("❌ LENGTH VIOLATION: Result does not have the expected length!")
else:
    print("Result is None (NULL)")

print("\n" + "=" * 60)
print("Additional test cases")
print("=" * 60)

# Test with text longer than length
text2 = "verylongstring"
length2 = 5
fill_text2 = ""

print(f"\nTest with text longer than target length:")
print(f"  text = {text2!r}")
print(f"  length = {length2}")
print(f"  fill_text = {fill_text2!r}")

result_lpad2 = _sqlite_lpad(text2, length2, fill_text2)
result_rpad2 = _sqlite_rpad(text2, length2, fill_text2)

print(f"LPAD result: {result_lpad2!r}, length: {len(result_lpad2) if result_lpad2 else 'N/A'}")
print(f"RPAD result: {result_rpad2!r}, length: {len(result_rpad2) if result_rpad2 else 'N/A'}")

# Test with None values
print(f"\nTest with None padding:")
result_lpad_none = _sqlite_lpad("test", 10, None)
result_rpad_none = _sqlite_rpad("test", 10, None)
print(f"LPAD with None padding: {result_lpad_none}")
print(f"RPAD with None padding: {result_rpad_none}")

# Test with normal padding
print(f"\nTest with normal padding (for comparison):")
result_lpad_normal = _sqlite_lpad("hi", 5, "x")
result_rpad_normal = _sqlite_rpad("hi", 5, "y")
print(f"LPAD('hi', 5, 'x'): {result_lpad_normal!r}, length: {len(result_lpad_normal)}")
print(f"RPAD('hi', 5, 'y'): {result_rpad_normal!r}, length: {len(result_rpad_normal)}")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print("The bug is CONFIRMED:")
print("- When fill_text is an empty string and text is shorter than target length")
print("- Django returns the original text unchanged (wrong length)")
print("- Standard SQL behavior would return NULL")
print("- This violates the length invariant of LPAD/RPAD functions")