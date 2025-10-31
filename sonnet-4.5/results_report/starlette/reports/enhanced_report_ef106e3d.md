# Bug Report: starlette.middleware.cors.CORSMiddleware Duplicate Headers in allow_headers List

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware creates duplicate headers in its internal `self.allow_headers` list when users provide headers that match CORS safelisted headers but with different casing (e.g., providing "accept" when "Accept" is already in SAFELISTED_HEADERS).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.cors import CORSMiddleware, SAFELISTED_HEADERS


def dummy_app(scope, receive, send):
    pass


@given(st.lists(st.sampled_from(list(SAFELISTED_HEADERS)), min_size=1, max_size=3))
def test_cors_no_duplicate_headers_when_user_provides_safelisted(safelisted_subset):
    user_headers = [h.lower() for h in safelisted_subset]

    cors = CORSMiddleware(
        dummy_app,
        allow_origins=["*"],
        allow_headers=user_headers,
        allow_methods=["GET"]
    )

    assert len(cors.allow_headers) == len(set(cors.allow_headers)), \
        f"Duplicate headers: {cors.allow_headers}"


if __name__ == "__main__":
    test_cors_no_duplicate_headers_when_user_provides_safelisted()
```

<details>

<summary>
**Failing input**: `safelisted_subset=['Content-Language']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 25, in <module>
    test_cors_no_duplicate_headers_when_user_provides_safelisted()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 10, in test_cors_no_duplicate_headers_when_user_provides_safelisted
    def test_cors_no_duplicate_headers_when_user_provides_safelisted(safelisted_subset):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 20, in test_cors_no_duplicate_headers_when_user_provides_safelisted
    assert len(cors.allow_headers) == len(set(cors.allow_headers)), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Duplicate headers: ['accept', 'accept-language', 'content-language', 'content-type', 'content-language']
Falsifying example: test_cors_no_duplicate_headers_when_user_provides_safelisted(
    safelisted_subset=['Content-Language'],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware


def dummy_app(scope, receive, send):
    pass


cors = CORSMiddleware(
    dummy_app,
    allow_origins=["*"],
    allow_headers=["accept"],
    allow_methods=["GET"]
)

print("allow_headers:", cors.allow_headers)
print("Expected: ['accept', 'accept-language', 'content-language', 'content-type']")
print("Actual has duplicate 'accept':", cors.allow_headers)
print("Length of allow_headers:", len(cors.allow_headers))
print("Length of set(allow_headers):", len(set(cors.allow_headers)))
assert len(cors.allow_headers) == len(set(cors.allow_headers)), f"Duplicate headers found: {cors.allow_headers}"
```

<details>

<summary>
AssertionError: Duplicate 'accept' header in allow_headers list
</summary>
```
allow_headers: ['accept', 'accept-language', 'content-language', 'content-type', 'accept']
Expected: ['accept', 'accept-language', 'content-language', 'content-type']
Actual has duplicate 'accept': ['accept', 'accept-language', 'content-language', 'content-type', 'accept']
Length of allow_headers: 5
Length of set(allow_headers): 4
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/repo.py", line 20, in <module>
    assert len(cors.allow_headers) == len(set(cors.allow_headers)), f"Duplicate headers found: {cors.allow_headers}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Duplicate headers found: ['accept', 'accept-language', 'content-language', 'content-type', 'accept']
```
</details>

## Why This Is A Bug

The CORSMiddleware constructor merges user-provided headers with SAFELISTED_HEADERS (defined on line 12 as `{"Accept", "Accept-Language", "Content-Language", "Content-Type"}` with title-case) on line 58, then lowercases everything on line 67. This creates duplicates when users provide lowercase versions of safelisted headers.

The bug occurs because:
1. Line 58: `allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))` treats "Accept" and "accept" as different strings during the set union
2. Line 67: `self.allow_headers = [h.lower() for h in allow_headers]` converts both to "accept", creating duplicates

While this doesn't break CORS functionality (the membership check on line 129 still works correctly with duplicates), it violates the expected behavior that `self.allow_headers` should contain unique headers. This leads to:
- Unnecessary memory usage from duplicate strings
- Slightly slower membership checks on line 129
- Unexpected internal state that could cause issues if the code is modified in the future

## Relevant Context

The CORS specification (https://www.w3.org/TR/cors/) treats header names as case-insensitive. The Starlette implementation correctly handles this for functionality but fails to maintain a clean internal state.

The safelisted headers are defined in the CORS specification as headers that are always allowed in simple CORS requests. The current code properly includes them but creates duplicates when users redundantly specify them with different casing.

Code references:
- SAFELISTED_HEADERS definition: `/home/npc/pbt/agentic-pbt/envs/mcp_env/lib/python3.13/site-packages/starlette/middleware/cors.py:12`
- Header merging logic: `/home/npc/pbt/agentic-pbt/envs/mcp_env/lib/python3.13/site-packages/starlette/middleware/cors.py:58`
- Lowercase conversion: `/home/npc/pbt/agentic-pbt/envs/mcp_env/lib/python3.13/site-packages/starlette/middleware/cors.py:67`
- Header validation: `/home/npc/pbt/agentic-pbt/envs/mcp_env/lib/python3.13/site-packages/starlette/middleware/cors.py:129`

## Proposed Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -55,10 +55,11 @@ class CORSMiddleware:
                 "Access-Control-Max-Age": str(max_age),
             }
         )
-        allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
+        # Normalize headers to lowercase before merging to avoid duplicates
+        normalized_safelisted = {h.lower() for h in SAFELISTED_HEADERS}
+        normalized_user_headers = {h.lower() for h in allow_headers}
+        allow_headers = sorted(normalized_safelisted | normalized_user_headers)
         if allow_headers and not allow_all_headers:
             preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
         if allow_credentials:
@@ -64,7 +65,7 @@ class CORSMiddleware:
         self.app = app
         self.allow_origins = allow_origins
         self.allow_methods = allow_methods
-        self.allow_headers = [h.lower() for h in allow_headers]
+        self.allow_headers = list(allow_headers)
         self.allow_all_headers = allow_all_headers
         self.allow_all_headers = allow_all_headers
         self.preflight_explicit_allow_origin = preflight_explicit_allow_origin
```