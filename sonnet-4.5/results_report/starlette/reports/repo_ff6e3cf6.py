from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers


def dummy_app(scope, receive, send):
    pass


cors = CORSMiddleware(
    app=dummy_app,
    allow_origins=["https://example.com"],
    allow_headers=[" X-Custom-Header "],
)

headers = Headers(raw=[
    (b"origin", b"https://example.com"),
    (b"access-control-request-method", b"GET"),
    (b"access-control-request-headers", b"X-Custom-Header"),
])

response = cors.preflight_response(request_headers=headers)

print(f"Status: {response.status_code}")
print(f"Body: {response.body}")