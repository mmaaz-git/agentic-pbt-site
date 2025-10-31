import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import io
from django.core.servers.basehttp import ServerHandler


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

    # The test assumes negative values should be treated as 0
    if expected_length < 0:
        expected_length = 0

    print(f"Testing with content_length_value={content_length_value}")
    print(f"  Stream limit: {stream.limit}")
    print(f"  Expected limit: {expected_length}")

    if stream.limit == expected_length:
        print(f"  ✓ Test passes - limits match")
    else:
        print(f"  ✗ Test FAILS - Expected limit {expected_length}, got {stream.limit}")
        return False

    return True

# Test with the specific failing input mentioned in the bug report
print("=== Testing with specific failing input: -1 ===")
result = test_serverhandler_content_length_parsing(-1)

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

print("\n=== Testing with float -3.14 ===")
test_serverhandler_content_length_parsing(-3.14)

print("\n\n=== SUMMARY ===")
print("The bug is confirmed: Django accepts negative CONTENT_LENGTH values")
print("RFC 9110 Section 8.6 requires Content-Length to be a non-negative decimal integer")
print("Current behavior: Negative values are passed through to LimitedStream")
print("Expected behavior: Negative values should be treated as 0 (like invalid non-numeric values)")