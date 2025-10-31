#!/usr/bin/env python3
"""Property-based test that discovers the console EmailBackend bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import io
from hypothesis import given, strategies as st
from django.core.mail.backends.console import EmailBackend as ConsoleBackend
from unittest.mock import Mock

@given(st.lists(st.just(Mock(message=lambda: Mock(as_bytes=lambda: b'test', get_charset=lambda: None))), max_size=10))
def test_console_backend_return_type_invariant(messages):
    stream = io.StringIO()
    backend = ConsoleBackend(stream=stream, fail_silently=False)
    result = backend.send_messages(messages)

    assert result is not None, f"send_messages should return an integer, not None"
    assert isinstance(result, int), f"send_messages should return an integer"
    assert result == len(messages), f"send_messages should return message count"

# Run the test
if __name__ == "__main__":
    test_console_backend_return_type_invariant()