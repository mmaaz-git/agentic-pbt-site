#!/usr/bin/env python3
"""Test how the middleware handles different port combinations"""

from starlette.datastructures import URL


def test_various_ports():
    """Test how the middleware logic handles different port combinations"""

    test_cases = [
        # (scheme, port, expected_behavior)
        ("http", 80, "Should strip port 80 (standard for HTTP)"),
        ("http", 443, "Should keep port 443 (non-standard for HTTP)"),
        ("http", 8080, "Should keep port 8080 (non-standard)"),
        ("http", 8443, "Should keep port 8443 (non-standard)"),
        ("ws", 80, "Should strip port 80 (standard for WS)"),
        ("ws", 443, "Should keep port 443 (non-standard for WS)"),
    ]

    for scheme, port, expected in test_cases:
        scope = {
            "type": "http" if scheme == "http" else "websocket",
            "scheme": scheme,
            "server": ("example.com", port),
            "path": "/test",
            "query_string": b"",
            "headers": []
        }

        url = URL(scope=scope)

        # Simulate the middleware logic
        netloc = url.hostname if url.port in (80, 443) else url.netloc
        redirect_scheme = {"http": "https", "ws": "wss"}.get(scheme, scheme)
        result_url = url.replace(scheme=redirect_scheme, netloc=netloc)

        # Check if port is preserved
        port_in_result = f":{port}" in str(result_url)

        print(f"\nTest case: {scheme}://example.com:{port}")
        print(f"  Original URL: {url}")
        print(f"  Redirect URL: {result_url}")
        print(f"  Port preserved: {port_in_result}")
        print(f"  Expected: {expected}")

        # Determine if behavior matches expectation
        if "Should strip" in expected:
            correct = not port_in_result
        else:
            correct = port_in_result
        print(f"  Behavior correct?: {correct}")


def analyze_standard_ports():
    """Analyze what are the standard ports for each scheme"""
    print("\n" + "="*50)
    print("Standard port analysis:")
    print("="*50)

    print("\nAccording to RFC standards:")
    print("  HTTP standard port: 80")
    print("  HTTPS standard port: 443")
    print("  WS standard port: 80 (same as HTTP)")
    print("  WSS standard port: 443 (same as HTTPS)")

    print("\nMiddleware current logic:")
    print("  Strips ports: 80 and 443 (regardless of scheme)")

    print("\nCorrect behavior should be:")
    print("  For HTTP -> HTTPS redirect:")
    print("    - Strip port 80 (standard for HTTP)")
    print("    - Keep port 443 (non-standard for HTTP)")
    print("  For WS -> WSS redirect:")
    print("    - Strip port 80 (standard for WS)")
    print("    - Keep port 443 (non-standard for WS)")


if __name__ == "__main__":
    test_various_ports()
    analyze_standard_ports()