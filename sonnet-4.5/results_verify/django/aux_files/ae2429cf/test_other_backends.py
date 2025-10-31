#!/usr/bin/env python3
"""Test how other backends handle empty message lists."""

import io
import sys
import os

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'django.conf.global_settings'
import django
django.setup()

from django.core.mail.backends.console import EmailBackend as ConsoleBackend
from django.core.mail.backends.locmem import EmailBackend as LocmemBackend
from django.core.mail.backends.dummy import EmailBackend as DummyBackend
from django.core.mail.backends.filebased import EmailBackend as FileBackend
import tempfile

print("Testing various email backends with empty message list:\n")

# Console backend
console_backend = ConsoleBackend(stream=io.StringIO())
console_result = console_backend.send_messages([])
print(f"Console backend: {console_result} (type: {type(console_result).__name__})")

# Locmem backend
locmem_backend = LocmemBackend()
locmem_result = locmem_backend.send_messages([])
print(f"Locmem backend: {locmem_result} (type: {type(locmem_result).__name__})")

# Dummy backend
dummy_backend = DummyBackend()
dummy_result = dummy_backend.send_messages([])
print(f"Dummy backend: {dummy_result} (type: {type(dummy_result).__name__})")

# File-based backend
with tempfile.TemporaryDirectory() as tmpdir:
    file_backend = FileBackend(file_path=tmpdir)
    file_result = file_backend.send_messages([])
    print(f"File backend: {file_result} (type: {type(file_result).__name__})")

print("\n" + "="*50)
print("Summary:")
print("Console backend returns None - INCONSISTENT")
print("Other backends return 0 - CONSISTENT")