# Bug Report: starlette.middleware.cors.CORSMiddleware Inconsistent Header Ordering with Different Case

**Target**: `starlette.middleware.cors.CORSMiddleware`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CORSMiddleware produces different internal `allow_headers` list orderings when the same HTTP header is provided with different capitalization, violating the HTTP specification principle that header names are case-insensitive.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.cors import CORSMiddleware


@given(
    header=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll"), whitelist_characters="-"))
)
def test_cors_headers_case_insensitive_property(header):
    middleware_upper = CORSMiddleware(None, allow_headers=[header.upper()])
    middleware_lower = CORSMiddleware(None, allow_headers=[header.lower()])

    assert middleware_upper.allow_headers == middleware_lower.allow_headers


# Run the test
if __name__ == "__main__":
    test_cors_headers_case_insensitive_property()
```

<details>

<summary>
**Failing input**: `header='A'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 17, in <module>
    test_cors_headers_case_insensitive_property()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 6, in test_cors_headers_case_insensitive_property
    header=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll"), whitelist_characters="-"))
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 12, in test_cors_headers_case_insensitive_property
    assert middleware_upper.allow_headers == middleware_lower.allow_headers
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_cors_headers_case_insensitive_property(
    header='A',
)
```
</details>

## Reproducing the Bug

```python
from starlette.middleware.cors import CORSMiddleware

# Create two middleware instances with the same header but different capitalization
middleware_upper = CORSMiddleware(None, allow_headers=['A'])
middleware_lower = CORSMiddleware(None, allow_headers=['a'])

print("With 'A':", middleware_upper.allow_headers)
print("With 'a':", middleware_lower.allow_headers)
print("Equal?", middleware_upper.allow_headers == middleware_lower.allow_headers)

# Additional test with another header
middleware_upper2 = CORSMiddleware(None, allow_headers=['X-Custom-Header'])
middleware_lower2 = CORSMiddleware(None, allow_headers=['x-custom-header'])

print("\nWith 'X-Custom-Header':", middleware_upper2.allow_headers)
print("With 'x-custom-header':", middleware_lower2.allow_headers)
print("Equal?", middleware_upper2.allow_headers == middleware_lower2.allow_headers)

# Test with multiple headers showing the sorting issue
middleware_multi_upper = CORSMiddleware(None, allow_headers=['B', 'A'])
middleware_multi_lower = CORSMiddleware(None, allow_headers=['b', 'a'])

print("\nWith ['B', 'A']:", middleware_multi_upper.allow_headers)
print("With ['b', 'a']:", middleware_multi_lower.allow_headers)
print("Equal?", middleware_multi_upper.allow_headers == middleware_multi_lower.allow_headers)
```

<details>

<summary>
Different internal header orderings based on input case
</summary>
```
With 'A': ['a', 'accept', 'accept-language', 'content-language', 'content-type']
With 'a': ['accept', 'accept-language', 'content-language', 'content-type', 'a']
Equal? False

With 'X-Custom-Header': ['accept', 'accept-language', 'content-language', 'content-type', 'x-custom-header']
With 'x-custom-header': ['accept', 'accept-language', 'content-language', 'content-type', 'x-custom-header']
Equal? True

With ['B', 'A']: ['a', 'accept', 'accept-language', 'b', 'content-language', 'content-type']
With ['b', 'a']: ['accept', 'accept-language', 'content-language', 'content-type', 'a', 'b']
Equal? False
```
</details>

## Why This Is A Bug

HTTP headers are case-insensitive according to RFC 7230 Section 3.2, which explicitly states: "Each header field consists of a case-insensitive field name followed by a colon...". The CORSMiddleware implementation acknowledges this principle by lowercasing headers (line 67), but the sorting operation occurs before the lowercasing (line 58), creating inconsistent internal state.

The issue arises from the interaction between Python's sorting behavior and ASCII values:
- Uppercase letters (A-Z) have ASCII values 65-90
- Lowercase letters (a-z) have ASCII values 97-122
- When sorting mixed-case strings, uppercase letters come before lowercase letters

This causes headers with uppercase letters to be sorted differently than their lowercase equivalents when combined with the SAFELISTED_HEADERS set (`{"Accept", "Accept-Language", "Content-Language", "Content-Type"}`), which uses title case.

While the actual CORS functionality remains correct (headers are checked case-insensitively in line 129), this inconsistency violates the metamorphic property that equivalent HTTP headers should produce equivalent internal states.

## Relevant Context

The bug occurs in `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/starlette/middleware/cors.py`:

- **Line 58**: `allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))` - Sorts headers with mixed case
- **Line 67**: `self.allow_headers = [h.lower() for h in allow_headers]` - Lowercases after sorting
- **Line 12**: `SAFELISTED_HEADERS = {"Accept", "Accept-Language", "Content-Language", "Content-Type"}` - Default headers in title case
- **Line 129**: Headers are correctly checked in a case-insensitive manner during request processing

The Starlette documentation does not explicitly document the case-sensitivity behavior, but given that HTTP headers are universally case-insensitive, users would reasonably expect `CORSMiddleware(allow_headers=['A'])` and `CORSMiddleware(allow_headers=['a'])` to behave identically.

## Proposed Fix

```diff
--- a/starlette/middleware/cors.py
+++ b/starlette/middleware/cors.py
@@ -55,12 +55,12 @@ class CORSMiddleware:
                 "Access-Control-Max-Age": str(max_age),
             }
         )
-        allow_headers = sorted(SAFELISTED_HEADERS | set(allow_headers))
+        lowercased_allow_headers = {h.lower() for h in allow_headers}
+        lowercased_safelisted = {h.lower() for h in SAFELISTED_HEADERS}
+        allow_headers = sorted(lowercased_safelisted | lowercased_allow_headers)
         if allow_headers and not allow_all_headers:
             preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
         if allow_credentials:
             preflight_headers["Access-Control-Allow-Credentials"] = "true"

         self.app = app
         self.allow_origins = allow_origins
         self.allow_methods = allow_methods
-        self.allow_headers = [h.lower() for h in allow_headers]
+        self.allow_headers = allow_headers
         self.allow_all_origins = allow_all_origins
         self.allow_all_headers = allow_all_headers
```