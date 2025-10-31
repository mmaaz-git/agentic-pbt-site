import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient
import asyncio

print("Testing TrustedHostMiddleware with IPv6 addresses")
print("=" * 60)

# Create a simple test application
app = Starlette()

@app.route("/")
def homepage(request):
    return PlainTextResponse("Hello")

# Test cases with different allowed hosts configurations
test_cases = [
    {
        "allowed_hosts": ["::1", "localhost"],
        "test_hosts": [
            ("[::1]:8000", "Expected to work with IPv6 localhost"),
            ("localhost:8000", "Expected to work with localhost"),
            ("[2001:db8::1]:8000", "Expected to fail - not in allowed list"),
        ]
    },
    {
        "allowed_hosts": ["2001:db8::1"],
        "test_hosts": [
            ("[2001:db8::1]:443", "Expected to work with specific IPv6"),
        ]
    }
]

for test_config in test_cases:
    print(f"\nTesting with allowed_hosts: {test_config['allowed_hosts']}")
    print("-" * 40)

    # Create middleware with specific allowed hosts
    middleware = TrustedHostMiddleware(app, allowed_hosts=test_config['allowed_hosts'])
    client = TestClient(middleware)

    for host_header, description in test_config['test_hosts']:
        print(f"\nHost header: {host_header}")
        print(f"Description: {description}")

        # Extract what the middleware will extract using its current logic
        extracted = host_header.split(":")[0]
        print(f"Middleware will extract: '{extracted}'")

        # Test the actual request
        try:
            response = client.get("/", headers={"host": host_header})
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")

            if response.status_code == 200:
                print("✓ Request accepted")
            else:
                print("✗ Request rejected")
        except Exception as e:
            print(f"Error during request: {e}")

print("\n" + "=" * 60)
print("Testing the proposed fix:")
print("=" * 60)

def parse_host_current(host_header):
    """Current implementation"""
    return host_header.split(":")[0]

def parse_host_proposed(host_header):
    """Proposed fix implementation"""
    if host_header.startswith("["):
        if "]:" in host_header:
            return host_header.split("]:")[0][1:]
        elif host_header.endswith("]"):
            return host_header[1:-1]
        else:
            return host_header
    else:
        return host_header.split(":")[0]

test_headers = [
    "[::1]:8000",
    "[2001:db8::1]:443",
    "[fe80::1]:80",
    "[::ffff:192.0.2.1]:8080",
    "[::1]",  # Without port
    "localhost:8000",
    "example.com:443",
    "192.168.1.1:80",
    "example.com",  # Without port
]

print("\nComparing current vs proposed parsing:")
print("-" * 60)
for header in test_headers:
    current = parse_host_current(header)
    proposed = parse_host_proposed(header)
    print(f"Header: {header:30}")
    print(f"  Current:  '{current}'")
    print(f"  Proposed: '{proposed}'")
    print(f"  Match: {current == proposed}")
    print()