from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import Headers

def dummy_app(scope, receive, send):
    pass

middleware = TrustedHostMiddleware(dummy_app, allowed_hosts=["::1"])

scope = {
    'type': 'http',
    'headers': [(b'host', b'[::1]:8080')]
}

headers = Headers(scope=scope)
host = headers.get("host", "").split(":")[0]

print(f"Host header: [::1]:8080")
print(f"Parsed host: {host}")
print(f"Expected: ::1")
print(f"Actual: [")

# Let's also test with more IPv6 addresses
test_cases = [
    "[::1]:8080",
    "[2001:db8::1]:80",
    "[fe80::1]:443"
]

for host_header in test_cases:
    scope = {
        'type': 'http',
        'headers': [(b'host', host_header.encode())]
    }
    headers = Headers(scope=scope)
    parsed = headers.get("host", "").split(":")[0]
    expected = host_header.split("]")[0][1:]
    print(f"\nHost header: {host_header}")
    print(f"Current parsing: {parsed}")
    print(f"Expected: {expected}")