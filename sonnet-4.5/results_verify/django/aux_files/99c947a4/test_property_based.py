import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st
import io
from django.core.servers.basehttp import ServerHandler


@given(st.one_of(
    st.none(),
    st.text(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.booleans(),
    st.lists(st.integers()),
))
def test_serverhandler_content_length_parsing(content_length_value):
    environ = {}
    if content_length_value is not None:
        environ["CONTENT_LENGTH"] = str(content_length_value)

    stdin = io.BytesIO(b"test data")
    stdout = io.BytesIO()
    stderr = io.BytesIO()

    handler = ServerHandler(stdin, stdout, stderr, environ)
    stream = handler.get_stdin()

    try:
        expected_length = int(content_length_value) if content_length_value is not None else 0
    except (ValueError, TypeError):
        expected_length = 0

    if expected_length < 0:
        expected_length = 0

    print(f"Testing with content_length_value={content_length_value}")
    print(f"  Stream limit: {stream.limit}")
    print(f"  Expected limit: {expected_length}")

    assert stream.limit == expected_length, f"Expected limit {expected_length}, got {stream.limit}"

# Test with the specific failing input mentioned in the bug report
print("=== Testing with specific failing input: -1 ===")
test_serverhandler_content_length_parsing(-1)

# Test with a few more negative values
print("\n=== Testing with -100 ===")
test_serverhandler_content_length_parsing(-100)

print("\n=== Testing with -5 ===")
test_serverhandler_content_length_parsing(-5)

print("\n=== Testing with 0 ===")
test_serverhandler_content_length_parsing(0)

print("\n=== Testing with positive 10 ===")
test_serverhandler_content_length_parsing(10)

print("\n=== Testing with None ===")
test_serverhandler_content_length_parsing(None)

print("\n=== Testing with 'invalid' string ===")
test_serverhandler_content_length_parsing('invalid')