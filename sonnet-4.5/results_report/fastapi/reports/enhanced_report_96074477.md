# Bug Report: fastapi_middleware_cors_duplicate_headers_2025-09-25 Case-Insensitive Header Duplication

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware incorrectly creates duplicate headers in its internal `allow_headers` list when users provide headers that differ only in case from the default safelisted headers (Accept, Accept-Language, Content-Language, Content-Type).

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from starlette.middleware.cors import CORSMiddleware

def dummy_app(scope, receive, send):
    pass

@given(st.lists(st.sampled_from(["Accept", "accept", "ACCEPT", "content-type", "Content-Type", "X-Custom-Header", "x-custom-header"]), max_size=10))
@settings(max_examples=500)
def test_cors_no_duplicate_headers(allow_headers):
    middleware = CORSMiddleware(dummy_app, allow_headers=allow_headers)

    unique_headers = set(middleware.allow_headers)
    assert len(middleware.allow_headers) == len(unique_headers), \
        f"Duplicate headers found! Input: {allow_headers}, Output: {middleware.allow_headers}"

if __name__ == "__main__":
    test_cors_no_duplicate_headers()
```

<details>

<summary>
**Failing input**: `['accept']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 20, in <module>
    test_cors_no_duplicate_headers()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 11, in test_cors_no_duplicate_headers
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 16, in test_cors_no_duplicate_headers
    assert len(middleware.allow_headers) == len(unique_headers), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Duplicate headers found! Input: ['accept'], Output: ['accept', 'accept-language', 'content-language', 'content-type', 'accept']
Falsifying example: test_cors_no_duplicate_headers(
    allow_headers=['accept'],
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from starlette.middleware.cors import CORSMiddleware

def dummy_app(scope, receive, send):
    pass

# Create middleware with lowercase "accept" header
middleware = CORSMiddleware(dummy_app, allow_headers=["accept"])

# Print the resulting allow_headers list
print("allow_headers:", middleware.allow_headers)

# Check for duplicates
unique_headers = set(middleware.allow_headers)
if len(middleware.allow_headers) != len(unique_headers):
    print(f"ERROR: Duplicate headers found!")
    print(f"  List length: {len(middleware.allow_headers)}")
    print(f"  Unique count: {len(unique_headers)}")

    # Find and print duplicates
    from collections import Counter
    header_counts = Counter(middleware.allow_headers)
    duplicates = {h: count for h, count in header_counts.items() if count > 1}
    print(f"  Duplicates: {duplicates}")
else:
    print("No duplicates found")
```

<details>

<summary>
Duplicate header 'accept' appears twice in allow_headers list
</summary>
```
allow_headers: ['accept', 'accept-language', 'content-language', 'content-type', 'accept']
ERROR: Duplicate headers found!
  List length: 5
  Unique count: 4
  Duplicates: {'accept': 2}
```
</details>

## Why This Is A Bug

This violates expected behavior because HTTP headers are case-insensitive according to RFC 7230 Section 3.2. When a user provides "accept" as an allowed header, they expect it to be treated as the same header as "Accept" from the safelisted headers. The current implementation creates duplicates because:

1. Line 58 in `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/starlette/middleware/cors.py` performs a set union between `SAFELISTED_HEADERS` (containing "Accept" with capital A) and the user-provided headers (containing "accept" with lowercase a), treating them as distinct values.

2. Line 67 then lowercases all headers, resulting in two "accept" entries in the final list.

This creates unnecessary redundancy in the `allow_headers` list and violates the principle of case-insensitive header handling. While the middleware still functions correctly, it performs redundant checks and creates an inconsistent internal state.

## Relevant Context

The bug occurs in the CORSMiddleware initialization process at lines 58 and 67 of `starlette/middleware/cors.py`:

- Line 12 defines: `SAFELISTED_HEADERS = {"Accept", "Accept-Language", "Content-Language", "Content-Type"}`
- Line 58: `allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))` - This creates a union but preserves case differences
- Line 67: `self.allow_headers = [h.lower() for h in allow_headers]` - This lowercases everything, creating duplicates

The middleware uses `self.allow_headers` to validate incoming CORS preflight requests at line 129. While duplicate checking doesn't break functionality, it creates unnecessary iterations and violates the expectation that headers should be unique and case-insensitive.

FastAPI re-exports this middleware directly from Starlette, so the bug affects both frameworks.

## Proposed Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -55,10 +55,11 @@ class CORSMiddleware:
                 "Access-Control-Max-Age": str(max_age),
             }
         )
-        allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
+        # Normalize all headers to lowercase before deduplication
+        normalized_headers = {h.lower() for h in SAFELISTED_HEADERS} | {h.lower() for h in allow_headers}
+        allow_headers = sorted(normalized_headers)
         if allow_headers and not allow_all_headers:
             preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
         if allow_credentials:
             preflight_headers["Access-Control-Allow-Credentials"] = "true"

@@ -64,7 +65,7 @@ class CORSMiddleware:
         self.app = app
         self.allow_origins = allow_origins
         self.allow_methods = allow_methods
-        self.allow_headers = [h.lower() for h in allow_headers]
+        self.allow_headers = list(allow_headers)
         self.allow_all_origins = allow_all_origins
         self.allow_all_headers = allow_all_headers
         self.preflight_explicit_allow_origin = preflight_explicit_allow_origin
```