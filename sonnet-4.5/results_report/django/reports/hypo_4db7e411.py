import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django
from django.conf import settings
if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

from hypothesis import given, strategies as st
import io
from django.core.servers.basehttp import ServerHandler

@given(st.integers(max_value=-1))
def test_serverhandler_rejects_negative_content_length(negative_length):
    environ = {"CONTENT_LENGTH": str(negative_length)}
    stdin = io.BytesIO(b"valid request body data")
    stdout = io.BytesIO()
    stderr = io.BytesIO()

    handler = ServerHandler(stdin, stdout, stderr, environ)

    # Property: Content-length should never be negative
    assert handler.stdin.limit >= 0, \
        f"LimitedStream.limit should be >= 0, got {handler.stdin.limit}"

    # Property: Valid data should be readable when present
    data = handler.stdin.read(10)
    assert len(data) > 0, "Should be able to read data when stream has content"

if __name__ == "__main__":
    test_serverhandler_rejects_negative_content_length()