#!/usr/bin/env python3
"""Test using hypothesis to reproduce the bug."""

import io
import sys
import os

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'django.conf.global_settings'
import django
django.setup()

from hypothesis import given, strategies as st
from django.core.mail.backends.console import EmailBackend as ConsoleBackend

@given(st.integers(min_value=0, max_value=10))
def test_console_backend_empty_list_returns_int(n):
    backend = ConsoleBackend(stream=io.StringIO())
    result = backend.send_messages([])
    assert isinstance(result, int), f"Expected int, got {type(result).__name__}: {result}"

# Run the test
print("Running hypothesis test...")
try:
    test_console_backend_empty_list_returns_int()
    print("All hypothesis tests passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")