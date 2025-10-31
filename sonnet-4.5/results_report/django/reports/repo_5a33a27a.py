from django.core.servers.basehttp import ServerHandler
from io import BytesIO

# Test case demonstrating the bug with negative Content-Length
stdin = BytesIO(b"request body data")
stdout = BytesIO()
stderr = BytesIO()
environ = {"CONTENT_LENGTH": "-1", "REQUEST_METHOD": "POST"}

handler = ServerHandler(stdin, stdout, stderr, environ)

print(f"LimitedStream limit: {handler.stdin.limit}")
print(f"Expected limit: 0 (should normalize negative to 0)")
print(f"Actual limit: {handler.stdin.limit}")

# Verify the bug exists
assert handler.stdin.limit == -1, f"Bug confirmed: negative limit {handler.stdin.limit} instead of 0"
print("\nBug confirmed: ServerHandler accepts negative Content-Length values")
print("This violates HTTP RFC 7230 which requires non-negative Content-Length")