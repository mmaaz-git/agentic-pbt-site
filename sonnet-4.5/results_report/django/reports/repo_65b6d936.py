#!/usr/bin/env python
"""
Minimal reproduction of Django validate_file_name backslash handling bug.
This demonstrates that the function incorrectly accepts filenames with
backslashes on Unix systems when allow_relative_path=False.
"""

import os
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.core.files.utils import validate_file_name
from django.core.exceptions import SuspiciousFileOperation

print("=== Django validate_file_name Backslash Handling Bug ===")
print(f"Platform: {os.name}")
print()

# Test case 1: Basic filename with backslash
print("Test 1: filename with backslash 'file\\name'")
print("  With allow_relative_path=False:")
try:
    result = validate_file_name('file\\name', allow_relative_path=False)
    print(f"    Result: {result!r} (ACCEPTED - This is the bug!)")
except SuspiciousFileOperation as e:
    print(f"    Raised exception: {e}")

print("  With allow_relative_path=True:")
try:
    result = validate_file_name('file\\name', allow_relative_path=True)
    print(f"    Result: {result!r}")
except SuspiciousFileOperation as e:
    print(f"    Raised exception: {e}")
print()

# Test case 2: Single backslash
print("Test 2: single backslash '\\\\'")
print("  With allow_relative_path=False:")
try:
    result = validate_file_name('\\', allow_relative_path=False)
    print(f"    Result: {result!r} (ACCEPTED - This is the bug!)")
except SuspiciousFileOperation as e:
    print(f"    Raised exception: {e}")
print()

# Test case 3: Path with multiple backslashes
print("Test 3: path with multiple backslashes 'path\\\\to\\\\file.txt'")
print("  With allow_relative_path=False:")
try:
    result = validate_file_name('path\\to\\file.txt', allow_relative_path=False)
    print(f"    Result: {result!r} (ACCEPTED - This is the bug!)")
except SuspiciousFileOperation as e:
    print(f"    Raised exception: {e}")
print()

# Test case 4: Compare with forward slash (should be rejected)
print("Test 4: forward slash 'dir/file' (for comparison)")
print("  With allow_relative_path=False:")
try:
    result = validate_file_name('dir/file', allow_relative_path=False)
    print(f"    Result: {result!r}")
except SuspiciousFileOperation as e:
    print(f"    Raised exception: {e} (CORRECTLY REJECTED)")
print()

# Demonstrate the os.path.basename behavior difference
print("=== Platform-specific os.path.basename behavior ===")
print(f"os.path.basename('file\\\\name') = {os.path.basename('file\\name')!r}")
print(f"os.path.basename('dir/file') = {os.path.basename('dir/file')!r}")
print()

print("=== Analysis ===")
print("On Unix systems, os.path.basename() doesn't treat backslash as a separator.")
print("This causes validate_file_name to accept backslashes when allow_relative_path=False,")
print("even though the allow_relative_path=True branch explicitly converts them to forward slashes.")
print("This inconsistency creates platform-dependent security behavior.")