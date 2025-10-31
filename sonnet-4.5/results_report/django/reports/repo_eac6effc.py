#!/usr/bin/env python3
"""
Minimal reproduction of Django console email backend bug.
This demonstrates that ConsoleBackend.send_messages() returns None
instead of 0 when called with an empty list of messages.
"""

import io
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.core.mail.backends.console import EmailBackend

# Create backend with StringIO to capture output
backend = EmailBackend(stream=io.StringIO())

# Call send_messages with empty list
result = backend.send_messages([])

# Print results
print(f"Result: {result}")
print(f"Type: {type(result).__name__}")

# Verify the bug
try:
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    assert result == 0, f"Expected 0, got {result}"
    print("✓ Test passed")
except AssertionError as e:
    print(f"✗ Assertion failed: {e}")

# Also test with other backends for comparison
print("\n--- Comparison with other backends ---")

from django.core.mail.backends.dummy import EmailBackend as DummyBackend
from django.core.mail.backends.locmem import EmailBackend as LocmemBackend

dummy_backend = DummyBackend()
dummy_result = dummy_backend.send_messages([])
print(f"Dummy backend: {dummy_result} (type: {type(dummy_result).__name__})")

locmem_backend = LocmemBackend()
locmem_result = locmem_backend.send_messages([])
print(f"Locmem backend: {locmem_result} (type: {type(locmem_result).__name__})")