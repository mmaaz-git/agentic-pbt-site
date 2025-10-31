# Bug Report: starlette.staticfiles.StaticFiles.is_not_modified Weak ETag Comparison Fails

**Target**: `starlette.staticfiles.StaticFiles.is_not_modified`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_not_modified()` method incorrectly fails to match weak ETags because it only normalizes the request's If-None-Match ETags but not the response's ETag header, causing identical weak ETags to be treated as different and breaking HTTP caching.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.datastructures import Headers
from starlette.staticfiles import StaticFiles

@given(st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=32, max_codepoint=126)).filter(lambda x: '"' not in x and ',' not in x))
def test_etag_weak_and_strong_match(etag_value):
    # Test that a weak ETag matches itself
    response_headers = Headers({"etag": f'W/"{etag_value}"'})
    request_headers = Headers({"if-none-match": f'W/"{etag_value}"'})

    static_files = StaticFiles(directory="/tmp", check_dir=False)
    result = static_files.is_not_modified(response_headers, request_headers)

    assert result is True, f"Weak ETag W/\"{etag_value}\" should match itself"

if __name__ == "__main__":
    # Run the test
    test_etag_weak_and_strong_match()
```

<details>

<summary>
**Failing input**: `etag_value='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 18, in <module>
    test_etag_weak_and_strong_match()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 6, in test_etag_weak_and_strong_match
    def test_etag_weak_and_strong_match(etag_value):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 14, in test_etag_weak_and_strong_match
    assert result is True, f"Weak ETag W/\"{etag_value}\" should match itself"
           ^^^^^^^^^^^^^^
AssertionError: Weak ETag W/"0" should match itself
Falsifying example: test_etag_weak_and_strong_match(
    etag_value='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from starlette.staticfiles import StaticFiles
from starlette.datastructures import Headers

# Create StaticFiles instance - using /tmp as it exists on all systems
static_files = StaticFiles(directory="/tmp", check_dir=False)

# Test 1: Weak ETag should match itself
response_headers = Headers({"etag": 'W/"123"'})
request_headers = Headers({"if-none-match": 'W/"123"'})
result = static_files.is_not_modified(response_headers, request_headers)
print(f"Test 1 - Weak ETag W/\"123\" matches itself: {result}")
print(f"  Expected: True, Got: {result}")

# Test 2: Strong ETag should match itself
response_headers2 = Headers({"etag": '"456"'})
request_headers2 = Headers({"if-none-match": '"456"'})
result2 = static_files.is_not_modified(response_headers2, request_headers2)
print(f"\nTest 2 - Strong ETag \"456\" matches itself: {result2}")
print(f"  Expected: True, Got: {result2}")

# Test 3: Multiple ETags with weak ETag
response_headers3 = Headers({"etag": 'W/"789"'})
request_headers3 = Headers({"if-none-match": '"123", W/"789", "456"'})
result3 = static_files.is_not_modified(response_headers3, request_headers3)
print(f"\nTest 3 - Weak ETag W/\"789\" in list: {result3}")
print(f"  Expected: True, Got: {result3}")

# Demonstrate the problem in the code
print("\n--- Debugging the issue ---")
etag = 'W/"123"'
if_none_match = 'W/"123"'
tags = [tag.strip(" W/") for tag in if_none_match.split(",")]
print(f"Response ETag: {etag}")
print(f"Request If-None-Match: {if_none_match}")
print(f"Stripped request tags: {tags}")
print(f"Is '{etag}' in {tags}? {etag in tags}")
print("\nThe bug: response ETag 'W/\"123\"' is never normalized, but request tags are.")
print("So we compare 'W/\"123\"' against ['\"123\"'], which fails!")
```

<details>

<summary>
Weak ETags fail to match themselves, while strong ETags work correctly
</summary>
```
Test 1 - Weak ETag W/"123" matches itself: False
  Expected: True, Got: False

Test 2 - Strong ETag "456" matches itself: True
  Expected: True, Got: True

Test 3 - Weak ETag W/"789" in list: False
  Expected: True, Got: False

--- Debugging the issue ---
Response ETag: W/"123"
Request If-None-Match: W/"123"
Stripped request tags: ['"123"']
Is 'W/"123"' in ['"123"']? False

The bug: response ETag 'W/"123"' is never normalized, but request tags are.
So we compare 'W/"123"' against ['"123"'], which fails!
```
</details>

## Why This Is A Bug

This violates HTTP caching standards defined in RFC 7232 Section 2.3.2, which mandates that for If-None-Match header processing, ETags must be compared using the "weak comparison function": two entity-tags are equivalent if their opaque-tags match character-by-character, regardless of either or both being tagged as 'weak'.

The bug occurs in line 207 of `/home/npc/miniconda/lib/python3.13/site-packages/starlette/staticfiles.py`:

```python
if etag in [tag.strip(" W/") for tag in if_none_match.split(",")]:
    return True
```

The code asymmetrically normalizes only the request's If-None-Match tags by stripping "W/" but leaves the response's ETag unchanged. This causes `'W/"123"'` to be compared against `['"123"']`, which always fails.

Additionally, `.strip(" W/")` has a subtle bug: it strips individual characters (' ', 'W', '/') from both ends, not the literal prefix "W/". This means edge cases like `'W/"abc"//WW'` would incorrectly become `'"abc"'`.

## Relevant Context

According to RFC 7232, weak ETags are commonly used by web servers and CDNs when content is semantically equivalent but may have minor differences (like different compression or whitespace). The current implementation breaks caching for all weak ETags, resulting in:

1. Unnecessary network traffic as browsers re-download unchanged content
2. Increased server load from serving files that should be cached
3. Slower page loads for users due to missed cache opportunities
4. Violation of HTTP/1.1 standards that most web infrastructure expects

Documentation: [RFC 7232 Section 2.3.2](https://datatracker.ietf.org/doc/html/rfc7232#section-2.3.2)

Code location: `/home/npc/miniconda/lib/python3.13/site-packages/starlette/staticfiles.py:207`

## Proposed Fix

```diff
--- a/starlette/staticfiles.py
+++ b/starlette/staticfiles.py
@@ -204,7 +204,13 @@ class StaticFiles:
         try:
             if_none_match = request_headers["if-none-match"]
             etag = response_headers["etag"]
-            if etag in [tag.strip(" W/") for tag in if_none_match.split(",")]:
+            # Normalize both response and request ETags for weak comparison
+            # Per RFC 7232, weak ETags (W/"value") should match when values are identical
+            response_etag_normalized = etag.strip().removeprefix("W/").strip()
+            request_etags_normalized = [
+                tag.strip().removeprefix("W/").strip()
+                for tag in if_none_match.split(",")
+            ]
+            if response_etag_normalized in request_etags_normalized:
                 return True
         except KeyError:
             pass
```