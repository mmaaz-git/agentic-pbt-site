#!/usr/bin/env python3
"""Minimal demonstration of the console EmailBackend bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import io
from django.core.mail.backends.console import EmailBackend

# Create a console backend with a string stream
stream = io.StringIO()
backend = EmailBackend(stream=stream)

# Call send_messages with an empty list
result = backend.send_messages([])

# Show the result
print(f"Result from send_messages([]): {result!r}")
print(f"Expected result: 0")
print(f"Type of result: {type(result)}")

# Test that it equals 0 (this will fail)
try:
    assert result == 0, f"Expected 0 but got {result!r}"
    print("PASS: Assertion succeeded")
except AssertionError as e:
    print(f"FAIL: {e}")

# Also demonstrate the issue with arithmetic operations
try:
    total = 0
    total += result  # This will fail with TypeError if result is None
    print(f"Arithmetic test passed: 0 + result = {total}")
except TypeError as e:
    print(f"FAIL: Arithmetic operation failed with TypeError: {e}")