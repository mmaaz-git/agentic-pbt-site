from starlette.middleware.wsgi import build_environ

scope = {
    "type": "http",
    "method": "GET",
    "path": "/test",
    "query_string": b"",
    "headers": [],
    "server": ("example.com", 8080),
    "http_version": "1.1",
}

environ = build_environ(scope, b"")
print(f"SERVER_PORT: {environ['SERVER_PORT']!r}")
print(f"Type: {type(environ['SERVER_PORT'])}")
print(f"Is string: {isinstance(environ['SERVER_PORT'], str)}")
print(f"PEP 3333 requires SERVER_PORT to be a string, but got {type(environ['SERVER_PORT']).__name__}")