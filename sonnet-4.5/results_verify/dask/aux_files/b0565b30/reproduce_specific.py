from starlette.middleware.cors import CORSMiddleware

async def dummy_app(scope, receive, send):
    pass

# Test the specific example from the bug report
middleware = CORSMiddleware(
    dummy_app,
    allow_origins=["http://example.com"],
    allow_headers=["X-Custom-Header ", "Content-Type"]
)

print("Stored allow_headers:", middleware.allow_headers)
print()

# Show what happens with whitespace headers
test_cases = [
    "X-Custom-Header ",
    " X-Custom-Header",
    " X-Custom-Header ",
    "X-Custom-Header",
]

for header in test_cases:
    stored = header.lower() if header in ["X-Custom-Header ", "Content-Type"] else header.lower()
    print(f"Input: '{header}'")
    print(f"After .lower(): '{header.lower()}'")
    print(f"After .lower().strip(): '{header.lower().strip()}'")
    if header.lower() in middleware.allow_headers:
        print(f"  -> Would be found in allow_headers")
    else:
        print(f"  -> Would NOT be found in allow_headers")
    print()

# Now let's simulate what happens in preflight_response
print("=" * 50)
print("Simulating preflight_response validation:")
print("=" * 50)

# Simulate a request with "X-Custom-Header" (no whitespace)
requested_headers = "X-Custom-Header"
print(f"Requested headers: '{requested_headers}'")

for header in [h.lower() for h in requested_headers.split(",")]:
    print(f"Processing header: '{header}'")
    print(f"After strip: '{header.strip()}'")
    if header.strip() not in middleware.allow_headers:
        print(f"ERROR: '{header.strip()}' not in allow_headers!")
        print(f"allow_headers contains: {middleware.allow_headers}")
    else:
        print(f"OK: '{header.strip()}' found in allow_headers")