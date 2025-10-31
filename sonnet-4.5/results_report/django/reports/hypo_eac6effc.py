#!/usr/bin/env python3
"""
Property-based test using Hypothesis to verify that Django's ConsoleBackend
always returns an integer from send_messages(), even with empty lists.
"""

import sys
import io
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from django.core.mail.backends.console import EmailBackend as ConsoleBackend

@given(st.integers(min_value=0, max_value=10))
def test_console_backend_empty_list_returns_int(n):
    """Test that ConsoleBackend.send_messages([]) returns an integer."""
    backend = ConsoleBackend(stream=io.StringIO())
    result = backend.send_messages([])
    assert isinstance(result, int), f"Expected int, got {type(result).__name__}: {result}"
    assert result == 0, f"Expected 0 for empty list, got {result}"

# Run the test
if __name__ == "__main__":
    test_console_backend_empty_list_returns_int()