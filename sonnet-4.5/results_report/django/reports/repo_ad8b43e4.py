from django.core.files.utils import validate_file_name
from django.core.exceptions import SuspiciousFileOperation

# Test case 1: Backslash path separator bypass
dangerous_name = "uploads\\..\\passwords.txt"
print(f"Testing dangerous name: {dangerous_name!r}")
try:
    result = validate_file_name(dangerous_name, allow_relative_path=False)
    print(f"BUG: Allowed dangerous name: {result!r}")
except SuspiciousFileOperation as e:
    print(f"Correctly rejected: {e}")

print()

# Test case 2: Forward slash is correctly blocked
safe_check = "uploads/../passwords.txt"
print(f"Testing with forward slash: {safe_check!r}")
try:
    result = validate_file_name(safe_check, allow_relative_path=False)
    print(f"BUG: Allowed dangerous name: {result!r}")
except SuspiciousFileOperation as e:
    print(f"Correctly rejected: {e}")

print()

# Test case 3: Simple backslash case
simple_backslash = "0\\file"
print(f"Testing simple backslash: {simple_backslash!r}")
try:
    result = validate_file_name(simple_backslash, allow_relative_path=False)
    print(f"BUG: Allowed name with backslash: {result!r}")
except SuspiciousFileOperation as e:
    print(f"Correctly rejected: {e}")

print()

# Test case 4: Demonstrate platform-dependent behavior of os.path.basename
import os
print("Platform-dependent behavior of os.path.basename:")
print(f"os.path.basename('a\\\\b') = {os.path.basename('a\\b')!r}")
print(f"os.path.basename('a/b') = {os.path.basename('a/b')!r}")