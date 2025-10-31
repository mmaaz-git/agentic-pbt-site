# Bug Report: CORSMiddleware allow_headers Contains Duplicates and Incorrect Sort Order

**Target**: `starlette.middleware.cors.CORSMiddleware.__init__`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware's `allow_headers` attribute contains duplicate headers when inputs differ only in case (e.g., 'Q' and 'q'), and headers are not properly sorted alphabetically due to operations occurring before case normalization.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from starlette.middleware.cors import CORSMiddleware

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=20)))
@settings(max_examples=1000)
@example(['Q', 'q'])  # Known failing example
def test_cors_allow_headers_no_duplicates(allow_headers):
    async def dummy_app(scope, receive, send):
        pass

    middleware = CORSMiddleware(app=dummy_app, allow_headers=allow_headers)
    assert len(middleware.allow_headers) == len(set(middleware.allow_headers))

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=20)))
@settings(max_examples=1000)
@example(['['])  # Known failing example
def test_cors_allow_headers_sorted(allow_headers):
    async def dummy_app(scope, receive, send):
        pass

    middleware = CORSMiddleware(app=dummy_app, allow_headers=allow_headers)
    assert middleware.allow_headers == sorted(middleware.allow_headers)

# Run the tests
if __name__ == "__main__":
    import traceback

    print("Running property-based tests for CORSMiddleware...")
    print("\nTest 1: Check for no duplicates")
    try:
        test_cors_allow_headers_no_duplicates()
        print("✓ No duplicates test passed")
    except Exception as e:
        print("✗ Duplicates test failed")
        traceback.print_exc()

    print("\nTest 2: Check for proper sorting")
    try:
        test_cors_allow_headers_sorted()
        print("✓ Sorting test passed")
    except Exception as e:
        print("✗ Sorting test failed")
        traceback.print_exc()
```

<details>

<summary>
**Failing input**: `['Q', 'q']` for duplicates test, `['[']` for sorting test
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 31, in <module>
    test_cors_allow_headers_no_duplicates()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 5, in test_cors_allow_headers_no_duplicates
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 12, in test_cors_allow_headers_no_duplicates
    assert len(middleware.allow_headers) == len(set(middleware.allow_headers))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying explicit example: test_cors_allow_headers_no_duplicates(
    allow_headers=['Q', 'q'],
)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 39, in <module>
    test_cors_allow_headers_sorted()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 15, in test_cors_allow_headers_sorted
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 22, in test_cors_allow_headers_sorted
    assert middleware.allow_headers == sorted(middleware.allow_headers)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying explicit example: test_cors_allow_headers_sorted(
    allow_headers=['['],
)
Running property-based tests for CORSMiddleware...

Test 1: Check for no duplicates
✗ Duplicates test failed

Test 2: Check for proper sorting
✗ Sorting test failed
```
</details>

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware

async def dummy_app(scope, receive, send):
    pass

# Test case 1: Duplicates bug
middleware = CORSMiddleware(app=dummy_app, allow_headers=['Q', 'q'])
print("Test 1 - Duplicates Bug:")
print("Input headers: ['Q', 'q']")
print("Result headers:", middleware.allow_headers)
print("Has duplicates:", len(middleware.allow_headers) != len(set(middleware.allow_headers)))
print("Expected: Headers should be deduplicated since 'Q' and 'q' are the same header (case-insensitive)")
print()

# Test case 2: Sorting bug
middleware2 = CORSMiddleware(app=dummy_app, allow_headers=['['])
print("Test 2 - Sorting Bug:")
print("Input headers: ['[']")
print("Result headers:", middleware2.allow_headers)
print("Is properly sorted:", middleware2.allow_headers == sorted(middleware2.allow_headers))
print("Expected: Headers should be in alphabetical order")
```

<details>

<summary>
Demonstrates duplicate 'q' headers and incorrect sort order with '[' character
</summary>
```
Test 1 - Duplicates Bug:
Input headers: ['Q', 'q']
Result headers: ['accept', 'accept-language', 'content-language', 'content-type', 'q', 'q']
Has duplicates: True
Expected: Headers should be deduplicated since 'Q' and 'q' are the same header (case-insensitive)

Test 2 - Sorting Bug:
Input headers: ['[']
Result headers: ['accept', 'accept-language', 'content-language', 'content-type', '[']
Is properly sorted: False
Expected: Headers should be in alphabetical order
```
</details>

## Why This Is A Bug

HTTP header field names are case-insensitive according to RFC 9110 Section 5.1, which states: "Field names are case-insensitive." Therefore, headers like 'Q' and 'q' should be treated as identical and deduplicated.

The code in `starlette/middleware/cors.py` explicitly attempts to:
1. **Deduplicate headers** using `set()` on line 58
2. **Sort headers alphabetically** using `sorted()` on line 58
3. **Normalize to lowercase** using `.lower()` on line 67

However, these operations occur in the wrong order. The deduplication and sorting happen BEFORE lowercasing, causing:
- Headers differing only in case (e.g., 'Q' and 'q') to be treated as distinct during deduplication, resulting in duplicates after lowercasing
- Sorting to be performed on mixed-case strings, but the final result contains only lowercase strings, breaking the alphabetical order (e.g., '[' sorts after 'Z' but before 'a' in ASCII)

This violates both the HTTP specification's requirement for case-insensitive header handling and the code's own documented intent to deduplicate and sort headers.

## Relevant Context

The bug occurs in the CORSMiddleware initialization at `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/starlette/middleware/cors.py:58-67`. The SAFELISTED_HEADERS constant contains `{"Accept", "Accept-Language", "Content-Language", "Content-Type"}` which are always included.

The middleware is widely used in FastAPI and Starlette applications for handling Cross-Origin Resource Sharing (CORS) policies. While the bug doesn't cause crashes or security issues, it results in incorrect CORS header handling that violates HTTP standards.

Documentation:
- Starlette CORS Middleware: https://www.starlette.io/middleware/#corsmiddleware
- HTTP Semantics RFC 9110: https://www.rfc-editor.org/rfc/rfc9110#section-5.1

## Proposed Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -55,10 +55,11 @@ class CORSMiddleware:
                 "Access-Control-Max-Age": str(max_age),
             }
         )
-        allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
+        # Normalize headers to lowercase before deduplication and sorting
+        allow_headers = sorted({h.lower() for h in SAFELISTED_HEADERS} | {h.lower() for h in allow_headers})
         if allow_headers and not allow_all_headers:
             preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
         if allow_credentials:
@@ -64,7 +65,7 @@ class CORSMiddleware:
         self.app = app
         self.allow_origins = allow_origins
         self.allow_methods = allow_methods
-        self.allow_headers = [h.lower() for h in allow_headers]
+        self.allow_headers = list(allow_headers)  # Already lowercase from above
         self.allow_all_origins = allow_all_origins
         self.allow_all_headers = allow_all_headers
         self.preflight_explicit_allow_origin = preflight_explicit_allow_origin
```