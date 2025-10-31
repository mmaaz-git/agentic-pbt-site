import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django
from django.conf import settings
if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

import io
from django.core.servers.basehttp import ServerHandler

# Test case with negative Content-Length
environ = {"CONTENT_LENGTH": "-100"}
stdin = io.BytesIO(b"valid request body data")
stdout = io.BytesIO()
stderr = io.BytesIO()

handler = ServerHandler(stdin, stdout, stderr, environ)

print(f"CONTENT_LENGTH: -100")
print(f"handler.stdin.limit: {handler.stdin.limit}")
print(f"Initial stdin position: {handler.stdin._pos}")

# Try to read data
data = handler.stdin.read(10)
print(f"Attempted to read 10 bytes")
print(f"Bytes read: {len(data)}")
print(f"Data: {data!r}")

# Verify that the data was lost even though stdin had content
stdin.seek(0)  # Reset to check original content was there
original_data = stdin.read()
print(f"\nOriginal stdin content (still present): {original_data!r}")
print(f"Length of original content: {len(original_data)} bytes")