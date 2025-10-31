# Bug Report: django.conf.urls.static Creates Unintended Catch-All URL Pattern with Slash-Only Prefix

**Target**: `django.conf.urls.static.static()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `static()` function incorrectly handles slash-only prefixes (e.g., "/", "//") by creating a catch-all URL pattern `^(?P<path>.*)$` that matches every possible URL, breaking Django's routing system and causing all requests to be handled as static file requests.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings as hyp_settings
from django.conf import settings
from django.conf.urls.static import static

settings.configure(DEBUG=True)

@hyp_settings(max_examples=200)
@given(st.integers(min_value=1, max_value=100))
def test_static_slash_only_prefix_bug(num_slashes):
    prefix = '/' * num_slashes
    result = static(prefix)

    if result:
        pattern_obj = result[0].pattern
        regex_pattern = pattern_obj.regex.pattern

        stripped = prefix.lstrip('/')

        if not stripped:
            assert regex_pattern != r'^(?P<path>.*)$', \
                f"BUG: Slash-only prefix '{prefix}' creates catch-all regex: {regex_pattern}"

# Run the test
test_static_slash_only_prefix_bug()
```

<details>

<summary>
**Failing input**: `num_slashes=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 24, in <module>
    test_static_slash_only_prefix_bug()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 8, in test_static_slash_only_prefix_bug
    @given(st.integers(min_value=1, max_value=100))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 20, in test_static_slash_only_prefix_bug
    assert regex_pattern != r'^(?P<path>.*)$', \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: BUG: Slash-only prefix '/' creates catch-all regex: ^(?P<path>.*)$
Falsifying example: test_static_slash_only_prefix_bug(
    num_slashes=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from django.conf import settings
from django.conf.urls.static import static

# Configure Django settings
settings.configure(DEBUG=True)

# Test with slash-only prefix
prefix = "/"
result = static(prefix)
pattern = result[0].pattern.regex

print(f"Prefix: '{prefix}'")
print(f"Stripped prefix: '{prefix.lstrip('/')}'")
print(f"Regex pattern: {pattern.pattern}")
print()

# Test what URLs this pattern matches
test_paths = ["media/file.jpg", "admin/login", "api/users", ""]
print("Testing URL matches:")
for path in test_paths:
    match = pattern.match(path)
    matches = match is not None
    print(f"  '{path}' matches: {matches}")
    if match:
        print(f"    Captured path: '{match.group('path')}'")
```

<details>

<summary>
Output showing catch-all pattern matching all URLs
</summary>
```
Prefix: '/'
Stripped prefix: ''
Regex pattern: ^(?P<path>.*)$

Testing URL matches:
  'media/file.jpg' matches: True
    Captured path: 'media/file.jpg'
  'admin/login' matches: True
    Captured path: 'admin/login'
  'api/users' matches: True
    Captured path: 'api/users'
  '' matches: True
    Captured path: ''
```
</details>

## Why This Is A Bug

This violates the expected behavior of the `static()` function which is designed to serve static files at a specific URL prefix location. The issue occurs because:

1. **Incomplete Input Validation**: Line 21 of `/django/conf/urls/static.py` checks `if not prefix:` to validate empty prefixes, but "/" is truthy in Python so it passes this check.

2. **Preprocessing Creates Invalid State**: Line 28 strips leading slashes with `prefix.lstrip("/")`, converting "/" into an empty string "".

3. **Unintended Pattern Generation**: When `re.escape("")` is called on the empty string, it returns "", resulting in the regex pattern `r"^%s(?P<path>.*)$" % ""` which becomes `^(?P<path>.*)$` - a pattern that matches ANY string.

4. **Silent Failure**: No error or warning is raised, making this bug extremely difficult to diagnose when it occurs.

5. **Complete Routing Breakdown**: The resulting catch-all pattern intercepts ALL URLs in the Django application, preventing normal view functions from being reached. Every request would be treated as a static file request.

The documentation examples always show meaningful prefixes like '/media/' or '/static/', never just '/'. While not explicitly documented as invalid, a slash-only prefix defeats the fundamental purpose of the function - to serve files under a specific prefix namespace.

## Relevant Context

The `static()` helper function is commonly used in Django development settings to serve user-uploaded media files or static assets. It's typically added to urlpatterns like this:

```python
urlpatterns = [
    # ... regular URL patterns ...
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

The function is intended for DEBUG mode only (line 23 checks `settings.DEBUG`), so this bug would primarily affect development environments. However, if someone misconfigures their production settings with DEBUG=True (a common mistake), this could affect production systems.

Django's URL routing works by checking patterns in order, so when a catch-all pattern is added, it prevents any subsequent patterns from being reached. This makes the bug particularly severe when triggered.

Documentation reference: https://docs.djangoproject.com/en/stable/ref/urls/#django.conf.urls.static.static

## Proposed Fix

The bug can be fixed by adding validation to reject prefixes that become empty after stripping slashes:

```diff
--- a/django/conf/urls/static.py
+++ b/django/conf/urls/static.py
@@ -20,6 +20,8 @@ def static(prefix, view=serve, **kwargs):
     """
     if not prefix:
         raise ImproperlyConfigured("Empty static prefix not permitted")
+    if prefix.lstrip("/") == "":
+        raise ImproperlyConfigured("Prefix cannot consist only of slashes")
     elif not settings.DEBUG or urlsplit(prefix).netloc:
         # No-op if not in debug mode or a non-local prefix.
         return []
```