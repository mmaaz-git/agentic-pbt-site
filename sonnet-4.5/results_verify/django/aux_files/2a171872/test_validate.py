#!/usr/bin/env python3
import os
import sys

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')
import django
from django.conf import settings
settings.configure(DEBUG=True)

from django.core.files.utils import validate_file_name
from django.core.exceptions import SuspiciousFileOperation

print("=== Testing validate_file_name function ===")
print(f"Platform: {os.name}")
print()

# Test 1: Basic reproduction from bug report
print("Test 1: Basic reproduction")
print("-" * 40)
filename_with_backslash = 'file\\name'
try:
    result = validate_file_name(filename_with_backslash, allow_relative_path=False)
    print(f"Input: {filename_with_backslash!r}")
    print(f"Result with allow_relative_path=False: {result!r}")
    print("No exception raised")
except SuspiciousFileOperation as e:
    print(f"Input: {filename_with_backslash!r}")
    print(f"Exception raised: {e}")

print()

# Test 2: Same filename with allow_relative_path=True
print("Test 2: Same filename with allow_relative_path=True")
print("-" * 40)
try:
    result = validate_file_name(filename_with_backslash, allow_relative_path=True)
    print(f"Input: {filename_with_backslash!r}")
    print(f"Result with allow_relative_path=True: {result!r}")
    print("No exception raised")
except SuspiciousFileOperation as e:
    print(f"Input: {filename_with_backslash!r}")
    print(f"Exception raised: {e}")

print()

# Test 3: Single backslash
print("Test 3: Single backslash")
print("-" * 40)
single_backslash = '\\'
try:
    result = validate_file_name(single_backslash, allow_relative_path=False)
    print(f"Input: {single_backslash!r}")
    print(f"Result with allow_relative_path=False: {result!r}")
    print("No exception raised")
except SuspiciousFileOperation as e:
    print(f"Input: {single_backslash!r}")
    print(f"Exception raised: {e}")

print()

# Test 4: Path with forward slashes
print("Test 4: Path with forward slashes")
print("-" * 40)
forward_slash_path = 'dir/file'
try:
    result = validate_file_name(forward_slash_path, allow_relative_path=False)
    print(f"Input: {forward_slash_path!r}")
    print(f"Result with allow_relative_path=False: {result!r}")
    print("No exception raised")
except SuspiciousFileOperation as e:
    print(f"Input: {forward_slash_path!r}")
    print(f"Exception raised: {e}")

print()

# Test 5: Understanding os.path.basename behavior
print("Test 5: os.path.basename behavior")
print("-" * 40)
test_cases = ['file\\name', 'dir/file', 'path\\to\\file', 'simple.txt', '\\', '/']
for test in test_cases:
    print(f"os.path.basename({test!r}) = {os.path.basename(test)!r}")

print()

# Test 6: Multiple backslashes
print("Test 6: Multiple backslashes")
print("-" * 40)
multi_backslash = 'path\\to\\file.txt'
try:
    result = validate_file_name(multi_backslash, allow_relative_path=False)
    print(f"Input: {multi_backslash!r}")
    print(f"Result with allow_relative_path=False: {result!r}")
    print("No exception raised")
except SuspiciousFileOperation as e:
    print(f"Input: {multi_backslash!r}")
    print(f"Exception raised: {e}")

print()

# Test 7: Mixed slashes
print("Test 7: Mixed forward and backslashes")
print("-" * 40)
mixed_slashes = 'dir/sub\\file.txt'
try:
    result = validate_file_name(mixed_slashes, allow_relative_path=False)
    print(f"Input: {mixed_slashes!r}")
    print(f"Result with allow_relative_path=False: {result!r}")
    print("No exception raised")
except SuspiciousFileOperation as e:
    print(f"Input: {mixed_slashes!r}")
    print(f"Exception raised: {e}")

print()

# Test 8: Check what the function does with allow_relative_path=True
print("Test 8: Conversion behavior with allow_relative_path=True")
print("-" * 40)
backslash_path = 'folder\\subfolder\\file.txt'
try:
    result = validate_file_name(backslash_path, allow_relative_path=True)
    print(f"Input: {backslash_path!r}")
    print(f"Result: {result!r}")
    print("No exception raised")
    print(f"Note: Backslashes were {'converted' if '\\' not in result else 'NOT converted'}")
except SuspiciousFileOperation as e:
    print(f"Input: {backslash_path!r}")
    print(f"Exception raised: {e}")