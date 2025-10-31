import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import io
from django.core.servers.basehttp import ServerHandler
from django.core.handlers.wsgi import WSGIRequest

# Test 1: Reproduce the bug with ServerHandler
print("=== Test 1: ServerHandler with negative CONTENT_LENGTH ===")
environ = {"CONTENT_LENGTH": "-5"}
stdin = io.BytesIO(b"12345")
stdout = io.BytesIO()
stderr = io.BytesIO()

handler = ServerHandler(stdin, stdout, stderr, environ)
stream = handler.get_stdin()

print(f"CONTENT_LENGTH: {environ['CONTENT_LENGTH']}")
print(f"LimitedStream limit: {stream.limit}")

data = stream.read()
print(f"Data read: {data}")
print(f"Expected: b'12345', Actual: {data}")
print()

# Test 2: Test with WSGIRequest
print("=== Test 2: WSGIRequest with negative CONTENT_LENGTH ===")
environ2 = {
    "CONTENT_LENGTH": "-10",
    "REQUEST_METHOD": "POST",
    "wsgi.input": io.BytesIO(b"Hello World"),
    "wsgi.url_scheme": "http"
}

request = WSGIRequest(environ2)
print(f"CONTENT_LENGTH: {environ2['CONTENT_LENGTH']}")
print(f"LimitedStream limit: {request._stream.limit}")

try:
    body_data = request.body
    print(f"Body data: {body_data}")
except Exception as e:
    print(f"Error reading body: {e}")

# Test 3: Check LimitedStream behavior with negative limit directly
print("\n=== Test 3: Direct LimitedStream test with negative limit ===")
from django.core.handlers.wsgi import LimitedStream

stream_input = io.BytesIO(b"test content")
limited = LimitedStream(stream_input, -5)
print(f"LimitedStream created with limit: {limited.limit}")
print(f"LimitedStream._pos: {limited._pos}")
print(f"Check if _pos >= limit: {limited._pos >= limited.limit} (0 >= -5 = True)")
data = limited.read()
print(f"Data read from negative-limited stream: {data}")
print(f"Expected empty bytes since 0 >= -5 is True")

# Test 4: Check with positive values for comparison
print("\n=== Test 4: Correct behavior with positive CONTENT_LENGTH ===")
environ3 = {"CONTENT_LENGTH": "5"}
stdin3 = io.BytesIO(b"12345678")
stdout3 = io.BytesIO()
stderr3 = io.BytesIO()

handler3 = ServerHandler(stdin3, stdout3, stderr3, environ3)
stream3 = handler3.get_stdin()

print(f"CONTENT_LENGTH: {environ3['CONTENT_LENGTH']}")
print(f"LimitedStream limit: {stream3.limit}")
data3 = stream3.read()
print(f"Data read: {data3}")
print(f"Expected: b'12345' (limited to 5 bytes), Actual: {data3}")