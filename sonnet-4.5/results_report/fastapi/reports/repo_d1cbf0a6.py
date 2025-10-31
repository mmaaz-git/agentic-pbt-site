from starlette.datastructures import URL

# Create a scope representing HTTP on port 443
scope = {
    "type": "http",
    "scheme": "http",
    "server": ("example.com", 443),
    "path": "/test",
    "query_string": b"",
    "headers": []
}

# Create URL from scope
url = URL(scope=scope)
print(f"Original URL: {url}")

# This is the exact logic from HTTPSRedirectMiddleware line 13-15
redirect_scheme = {"http": "https", "ws": "wss"}[url.scheme]
netloc = url.hostname if url.port in (80, 443) else url.netloc
result_url = url.replace(scheme=redirect_scheme, netloc=netloc)

print(f"Redirect URL: {result_url}")
print(f"\nProblem: Port 443 was stripped even though it's non-standard for HTTP!")
print(f"Expected: https://example.com:443/test")
print(f"Actual:   {result_url}")