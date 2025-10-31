from starlette.middleware.wsgi import build_environ

scope = {
    "type": "http",
    "method": "GET",
    "path": "/",
    "query_string": b"",
    "headers": [],
    "server": ("example.com", 8000),
    "http_version": "1.1",
}

environ = build_environ(scope, b"")

print(f"SERVER_PORT value: {repr(environ['SERVER_PORT'])}")
print(f"SERVER_PORT type: {type(environ['SERVER_PORT'])}")
print(f"Is SERVER_PORT a string? {isinstance(environ['SERVER_PORT'], str)}")

# According to PEP 3333, SERVER_PORT must be a string
# Let's verify if it's actually an integer
assert isinstance(environ['SERVER_PORT'], str), f"SERVER_PORT should be str, got {type(environ['SERVER_PORT'])}"