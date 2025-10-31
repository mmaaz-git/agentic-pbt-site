"""Reproduce the reported bug in starlette.datastructures.URL.replace"""

from hypothesis import given, strategies as st, settings
from starlette.datastructures import URL

print("Testing property-based test from bug report...")

@given(st.integers(min_value=1, max_value=65535))
@settings(max_examples=100)
def test_url_replace_port_should_not_crash(port):
    test_urls = [
        "http://@/path",
        "http:///path",
        "http://user@/path",
        "http://user:pass@/path",
    ]

    for url_str in test_urls:
        url = URL(url_str)
        try:
            new_url = url.replace(port=port)
            assert isinstance(new_url, URL)
            print(f"✓ URL({url_str!r}).replace(port={port}) succeeded: {new_url}")
        except IndexError as e:
            print(f"✗ URL({url_str!r}).replace(port={port}) failed with IndexError: {e}")
            raise

# Run the test
try:
    test_url_replace_port_should_not_crash()
    print("\nAll tests passed!")
except Exception as e:
    print(f"\nTest failed: {e}")

print("\n" + "="*60)
print("Testing manual reproduction from bug report...")

test_cases = [
    ("http://@/path", 8080),
    ("http:///path", 8080),
    ("http://user@/path", 8080),
    ("http://user:pass@/path", 8080),
]

for url_str, port in test_cases:
    print(f"\nTesting: URL({url_str!r}).replace(port={port})")
    try:
        url = URL(url_str)
        print(f"  Original URL properties:")
        print(f"    - netloc: {url.netloc!r}")
        print(f"    - hostname: {url.hostname!r}")
        print(f"    - username: {url.username!r}")
        print(f"    - port: {url.port!r}")

        new_url = url.replace(port=port)
        print(f"  ✓ Success! New URL: {new_url}")
        print(f"    - new netloc: {new_url.netloc!r}")
        print(f"    - new hostname: {new_url.hostname!r}")
        print(f"    - new port: {new_url.port!r}")
    except IndexError as e:
        print(f"  ✗ IndexError: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"  ✗ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("Examining the problematic code path...")

# Let's trace through what happens in the replace method
url_str = "http://@/path"
url = URL(url_str)
print(f"\nFor URL: {url_str!r}")
print(f"  netloc = {url.netloc!r}")
print(f"  hostname = {url.hostname!r}")

# Simulate what happens in replace() method when hostname is None
if url.hostname is None:
    netloc = url.netloc
    _, _, hostname = netloc.rpartition("@")
    print(f"  After rpartition('@'): hostname = {hostname!r}")
    print(f"  len(hostname) = {len(hostname)}")
    if hostname:
        print(f"  hostname[-1] would be: {hostname[-1]!r}")
    else:
        print(f"  hostname is empty string - accessing hostname[-1] would raise IndexError!")