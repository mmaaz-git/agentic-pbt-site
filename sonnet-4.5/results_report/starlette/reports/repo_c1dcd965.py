import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

# Demonstrate the actual bug in TrustedHostMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import Headers

# Simulate what happens in TrustedHostMiddleware at line 40
host_header = "[::1]:8000"
extracted_host = host_header.split(":")[0]

print(f"Host header: {host_header}")
print(f"Extracted host using split(':')[0]: {extracted_host}")
print(f"Expected host: ::1")
print()

# Test with various IPv6 addresses
test_cases = [
    "[::1]:8000",
    "[2001:db8::1]:443",
    "[fe80::1]:80",
    "[::ffff:192.0.2.1]:8080"
]

print("Testing various IPv6 addresses:")
for test_host in test_cases:
    extracted = test_host.split(":")[0]
    # Correct parsing would be:
    if test_host.startswith("[") and "]:" in test_host:
        expected = test_host.split("]:")[0][1:]
    elif test_host.startswith("[") and test_host.endswith("]"):
        expected = test_host[1:-1]
    else:
        expected = test_host.split(":")[0]

    print(f"  Input: {test_host}")
    print(f"    Current extraction: {extracted}")
    print(f"    Expected: {expected}")
    print(f"    Match: {extracted == expected}")
    print()

# Show that this causes real middleware failures
print("Real-world impact demonstration:")
print("Creating TrustedHostMiddleware with allowed_hosts=['::1']")

# Create a mock scope object
scope = {
    "type": "http",
    "headers": [(b"host", b"[::1]:8000")]
}

# Simulate the middleware logic
headers = Headers(scope=scope)
host = headers.get("host", "").split(":")[0]
allowed_hosts = ["::1"]

is_valid_host = False
for pattern in allowed_hosts:
    if host == pattern:
        is_valid_host = True
        break

print(f"  Host header received: [::1]:8000")
print(f"  Host extracted by middleware: {host}")
print(f"  Allowed hosts: {allowed_hosts}")
print(f"  Is valid host? {is_valid_host}")
print(f"  Result: {'Request would be ACCEPTED' if is_valid_host else 'Request would be REJECTED with 400 Invalid host header'}")

# Final assertion to demonstrate the bug
try:
    assert extracted_host == "::1", f"Bug confirmed: extracted '{extracted_host}' instead of '::1'"
except AssertionError as e:
    print(f"\n{e}")