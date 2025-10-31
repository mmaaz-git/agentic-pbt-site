#!/usr/bin/env python3
"""Test script to reproduce the reported bug."""

import io
import sys
import os

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'django.conf.global_settings'
import django
django.setup()

from django.core.mail.backends.console import EmailBackend

print("Testing console backend with empty message list...")
backend = EmailBackend(stream=io.StringIO())
result = backend.send_messages([])

print(f"Result: {result}")
print(f"Type: {type(result).__name__}")

try:
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    assert result == 0, f"Expected 0, got {result}"
    print("Test PASSED")
except AssertionError as e:
    print(f"Test FAILED: {e}")

print("\nTesting with one message...")
from django.core.mail import EmailMessage
msg = EmailMessage(
    'Hello',
    'Body goes here',
    'from@example.com',
    ['to@example.com'],
)
result2 = backend.send_messages([msg])
print(f"Result with 1 message: {result2}")
print(f"Type: {type(result2).__name__}")