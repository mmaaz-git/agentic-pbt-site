from hypothesis import given, strategies as st
from django.core.servers.basehttp import ServerHandler
from io import BytesIO

@given(st.integers())
def test_serverhandler_content_length_parsing_integers(content_length):
    stdin = BytesIO(b"test data")
    stdout = BytesIO()
    stderr = BytesIO()

    environ = {"CONTENT_LENGTH": str(content_length), "REQUEST_METHOD": "POST"}

    handler = ServerHandler(stdin, stdout, stderr, environ)

    expected_limit = max(0, content_length)
    assert handler.stdin.limit == expected_limit, f"Expected {expected_limit}, got {handler.stdin.limit}"

# Run the test
if __name__ == "__main__":
    test_serverhandler_content_length_parsing_integers()