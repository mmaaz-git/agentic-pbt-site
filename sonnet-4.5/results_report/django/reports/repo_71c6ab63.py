#!/usr/bin/env python3
"""Minimal reproduction of Django UploadedFile backslash vulnerability"""

import os
import sys

# Add Django to path if needed
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Setup minimal Django settings
import django
from django.conf import settings
settings.configure(DEBUG=True, SECRET_KEY='test')
django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile

# Test case 1: Basic backslash in filename
print("Test 1: Basic backslash in filename")
f = SimpleUploadedFile('test\\file.txt', b'content')
print(f"  Input: 'test\\\\file.txt'")
print(f"  Output: '{f.name}'")
print(f"  Contains backslash: {'\\' in f.name}")
print()

# Test case 2: Multiple backslashes (potential path traversal)
print("Test 2: Multiple backslashes (path traversal attempt)")
f2 = SimpleUploadedFile('..\\..\\etc\\passwd', b'malicious')
print(f"  Input: '..\\\\..\\\\etc\\\\passwd'")
print(f"  Output: '{f2.name}'")
print(f"  Contains backslash: {'\\' in f2.name}")
print()

# Test case 3: Mixed forward and backslashes
print("Test 3: Mixed forward and backslashes")
f3 = SimpleUploadedFile('../test\\..\\file.txt', b'mixed')
print(f"  Input: '../test\\\\..\\\\file.txt'")
print(f"  Output: '{f3.name}'")
print(f"  Contains backslash: {'\\' in f3.name}")
print(f"  Contains forward slash: {'/' in f3.name}")
print()

# Test case 4: Just a backslash
print("Test 4: Single backslash character")
f4 = SimpleUploadedFile('\\', b'')
print(f"  Input: '\\\\'")
print(f"  Output: '{f4.name}'")
print(f"  Name equals backslash: {f4.name == '\\\\'}")
print()

# Show os.path.basename behavior on current system
print("System information:")
print(f"  Operating System: {os.name}")
print(f"  os.path.basename('test\\\\file.txt') = '{os.path.basename('test\\\\file.txt')}'")
print(f"  os.path.basename('test/file.txt') = '{os.path.basename('test/file.txt')}'")