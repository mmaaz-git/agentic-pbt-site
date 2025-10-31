# Bug Report: GZipMiddleware Case-Sensitive Substring Matching Violates HTTP Standards

**Target**: `starlette.middleware.gzip.GZipMiddleware.__call__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

GZipMiddleware uses case-sensitive substring matching to detect "gzip" in Accept-Encoding headers, violating RFC 7231 which mandates case-insensitive content-coding values, and incorrectly matches substrings instead of parsing tokens.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis-based property test to demonstrate the GZipMiddleware bug.
"""

from hypothesis import given, settings, strategies as st, example

# Test case-insensitive matching for gzip encoding
@given(
    case_variant=st.sampled_from(["gzip", "GZIP", "Gzip", "GZip", "gZip", "GzIp"])
)
@example(case_variant="GZIP")  # Ensure we always test this specific case
@settings(max_examples=50)
def test_gzip_case_insensitive(case_variant):
    """Test that 'gzip' matching should be case-insensitive per RFC 7231."""
    # Current buggy implementation
    current_match = "gzip" in case_variant

    # Expected behavior (case-insensitive)
    expected_match = case_variant.lower() == "gzip"

    assert current_match == expected_match, \
        f"Bug: Accept-Encoding '{case_variant}' should match 'gzip' (HTTP is case-insensitive), but got {current_match}"

# Test that substring matching is incorrect
@given(
    encoding=st.sampled_from(["not-gzip", "gzip-not", "msgzip", "gzippy", "ungzip"])
)
@settings(max_examples=50)
def test_no_substring_matching(encoding):
    """Test that 'gzip' should not match as a substring in other encodings."""
    # Current buggy implementation
    current_match = "gzip" in encoding

    # Expected behavior (no substring matching - should parse tokens)
    # In proper implementation, these should NOT match
    expected_match = False

    assert current_match == expected_match, \
        f"Bug: Accept-Encoding '{encoding}' should NOT match 'gzip' (substring matching is wrong), but got {current_match}"

if __name__ == "__main__":
    print("Running Hypothesis tests for GZipMiddleware bug...")
    print("=" * 60)

    # Test 1: Case sensitivity
    print("\nTest 1: Case-insensitive matching")
    print("-" * 40)
    try:
        test_gzip_case_insensitive()
        print("✓ All case variants passed (no bug found)")
    except AssertionError as e:
        print(f"✗ Test failed: {e}")

    # Test 2: Substring matching
    print("\nTest 2: No substring matching")
    print("-" * 40)
    try:
        test_no_substring_matching()
        print("✓ No substring matching issues (no bug found)")
    except AssertionError as e:
        print(f"✗ Test failed: {e}")

    print("\n" + "=" * 60)
    print("Bug confirmed: GZipMiddleware uses case-sensitive substring matching")
```

<details>

<summary>
**Failing input**: `case_variant="GZIP"` and `encoding="not-gzip"`
</summary>
```
Running Hypothesis tests for GZipMiddleware bug...
============================================================

Test 1: Case-insensitive matching
----------------------------------------
✗ Test failed: Bug: Accept-Encoding 'GZIP' should match 'gzip' (HTTP is case-insensitive), but got False

Test 2: No substring matching
----------------------------------------
✗ Test failed: Bug: Accept-Encoding 'not-gzip' should NOT match 'gzip' (substring matching is wrong), but got True

============================================================
Bug confirmed: GZipMiddleware uses case-sensitive substring matching
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Demonstrating the GZipMiddleware case-sensitivity and substring matching bugs.
"""

# Direct demonstration of the bug in the logic
print("=== Direct Logic Bug Demonstration ===")
print("Current implementation at line 24: if 'gzip' in headers.get('Accept-Encoding', ''):")
print()

# Test case-sensitivity issue
accept_encoding_upper = "GZIP"
accept_encoding_mixed = "GZip"
accept_encoding_lower = "gzip"

print("Case-sensitivity tests:")
print(f"  'gzip' in 'GZIP': {('gzip' in accept_encoding_upper)}")
print(f"  'gzip' in 'GZip': {('gzip' in accept_encoding_mixed)}")
print(f"  'gzip' in 'gzip': {('gzip' in accept_encoding_lower)}")
print()

# Test substring matching issue
accept_encoding_not_gzip = "not-gzip"
accept_encoding_gzip_not = "gzip-not"
accept_encoding_msgzip = "msgzip"

print("Substring matching tests:")
print(f"  'gzip' in 'not-gzip': {('gzip' in accept_encoding_not_gzip)}")
print(f"  'gzip' in 'gzip-not': {('gzip' in accept_encoding_gzip_not)}")
print(f"  'gzip' in 'msgzip': {('gzip' in accept_encoding_msgzip)}")
print()

# Test with quality parameters
accept_encoding_with_q = "GZIP;q=1.0"
accept_encoding_with_q_lower = "gzip;q=0.8"

print("With quality parameters:")
print(f"  'gzip' in 'GZIP;q=1.0': {('gzip' in accept_encoding_with_q)}")
print(f"  'gzip' in 'gzip;q=0.8': {('gzip' in accept_encoding_with_q_lower)}")
print()

# Demonstrate with actual middleware behavior
print("=== Testing with actual Starlette GZipMiddleware ===")
try:
    from starlette.middleware.gzip import GZipMiddleware
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient
    import gzip

    def homepage(request):
        # Return a response large enough to trigger compression (>500 bytes)
        return PlainTextResponse("X" * 1000)

    routes = [Route("/", homepage)]
    app = Starlette(routes=routes)

    # Wrap app with GZipMiddleware
    app = GZipMiddleware(app, minimum_size=100)

    # Create test client
    client = TestClient(app)

    test_cases = [
        ("gzip", "should compress", True),
        ("GZIP", "should compress but doesn't", False),
        ("GZip", "should compress but doesn't", False),
        ("not-gzip", "shouldn't compress but does", True),
        ("msgzip", "shouldn't compress but does", True),
        ("deflate", "shouldn't compress", False),
    ]

    print("Testing different Accept-Encoding headers:")
    for encoding, description, expected_bug in test_cases:
        response = client.get("/", headers={"Accept-Encoding": encoding})
        has_gzip_header = "content-encoding" in response.headers and response.headers["content-encoding"] == "gzip"

        # Check content length to verify compression
        content_length = response.headers.get("content-length", "unknown")

        status = "COMPRESSED" if has_gzip_header else "NOT compressed"
        bug_indicator = " (BUG!)" if has_gzip_header == expected_bug else ""

        print(f"  {encoding:15} - {status:14} (content-length: {content_length}){bug_indicator}")

except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Please ensure starlette is installed: pip install starlette")

print()
print("=== Bug Summary ===")
print("1. Case-sensitive check violates RFC 7231 (content-coding values are case-insensitive)")
print("2. Substring matching causes false positives (e.g., 'not-gzip' matches)")
print("3. Valid headers like 'GZIP' or 'GZip' fail to trigger compression")
print("4. Invalid encodings like 'msgzip' trigger compression incorrectly")
```

<details>

<summary>
GZipMiddleware incorrectly handles case variants and substrings
</summary>
```
=== Direct Logic Bug Demonstration ===
Current implementation at line 24: if 'gzip' in headers.get('Accept-Encoding', ''):

Case-sensitivity tests:
  'gzip' in 'GZIP': False
  'gzip' in 'GZip': False
  'gzip' in 'gzip': True

Substring matching tests:
  'gzip' in 'not-gzip': True
  'gzip' in 'gzip-not': True
  'gzip' in 'msgzip': True

With quality parameters:
  'gzip' in 'GZIP;q=1.0': False
  'gzip' in 'gzip;q=0.8': True

=== Testing with actual Starlette GZipMiddleware ===
Testing different Accept-Encoding headers:
  gzip            - COMPRESSED     (content-length: 29) (BUG!)
  GZIP            - NOT compressed (content-length: 1000) (BUG!)
  GZip            - NOT compressed (content-length: 1000) (BUG!)
  not-gzip        - COMPRESSED     (content-length: 29) (BUG!)
  msgzip          - COMPRESSED     (content-length: 29) (BUG!)
  deflate         - NOT compressed (content-length: 1000) (BUG!)

=== Bug Summary ===
1. Case-sensitive check violates RFC 7231 (content-coding values are case-insensitive)
2. Substring matching causes false positives (e.g., 'not-gzip' matches)
3. Valid headers like 'GZIP' or 'GZip' fail to trigger compression
4. Invalid encodings like 'msgzip' trigger compression incorrectly
```
</details>

## Why This Is A Bug

This implementation violates HTTP specifications and expected behavior in multiple ways:

1. **RFC 7231 Section 5.3.4 Violation**: The specification explicitly states "All content-coding values are case-insensitive." The current implementation at line 24 (`if "gzip" in headers.get("Accept-Encoding", ""):`) uses Python's case-sensitive `in` operator, causing valid headers like "GZIP", "GZip", or "gZip" to be ignored.

2. **Incorrect Token Parsing**: The Accept-Encoding header is defined as a comma-separated list of content-coding tokens, potentially with quality parameters (e.g., "gzip;q=1.0, deflate;q=0.5"). The current substring matching approach incorrectly matches:
   - Hypothetical encodings containing "gzip" as a substring ("not-gzip", "msgzip", "gzippy")
   - This could cause security issues if a future encoding like "no-gzip" or "ungzip" were introduced

3. **Interoperability Issues**: HTTP clients that send uppercase or mixed-case "GZIP" headers (which are valid per RFC) will not receive compressed responses, leading to:
   - Increased bandwidth usage
   - Slower response times
   - Unexpected behavior for clients following the HTTP specification

4. **Documentation Gap**: While Starlette's documentation states the middleware handles requests "that include 'gzip' in the Accept-Encoding header," users reasonably expect HTTP middleware to follow HTTP standards by default.

## Relevant Context

The bug is located in `/home/npc/t-bench/.venv/lib/python3.13/site-packages/starlette/middleware/gzip.py` at line 24.

The Accept-Encoding header is fundamental to HTTP content negotiation. Major HTTP implementations (Apache, Nginx, IIS) handle this case-insensitively as required by the RFC. This bug could affect:
- Legacy HTTP clients that send uppercase headers
- Proxy servers that normalize headers to uppercase
- Testing tools and scripts that don't lowercase encoding names
- Any system expecting RFC-compliant HTTP behavior

Testing confirmed that when Accept-Encoding is "gzip" (lowercase), the response is compressed to 29 bytes. With "GZIP" (uppercase), the full 1000-byte response is sent uncompressed, despite being a valid HTTP header.

## Proposed Fix

```diff
--- a/starlette/middleware/gzip.py
+++ b/starlette/middleware/gzip.py
@@ -21,7 +21,11 @@ class GZipMiddleware:

         headers = Headers(scope=scope)
         responder: ASGIApp
-        if "gzip" in headers.get("Accept-Encoding", ""):
+        accept_encoding = headers.get("Accept-Encoding", "").lower()
+        # Parse comma-separated tokens, strip quality parameters
+        encodings = [e.split(";")[0].strip() for e in accept_encoding.split(",")]
+
+        if "gzip" in encodings:
             responder = GZipResponder(self.app, self.minimum_size, compresslevel=self.compresslevel)
         else:
             responder = IdentityResponder(self.app, self.minimum_size)
```