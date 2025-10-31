# Bug Report: Django TemplateCommand is_url Accepts Invalid Protocol-Only URLs

**Target**: `django.core.management.templates.TemplateCommand.is_url`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_url()` method in Django's TemplateCommand incorrectly returns `True` for protocol-only URLs like `"http://"`, `"https://"`, and `"ftp://"`. These are not valid downloadable URLs and produce filenames with colons (e.g., `"http:"`) which are invalid on Windows.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.core.management.templates import TemplateCommand

@given(st.sampled_from(["http", "https", "ftp"]))
@settings(max_examples=100)
def test_is_url_rejects_protocol_only_urls(protocol):
    cmd = TemplateCommand()
    protocol_only_url = f"{protocol}://"

    if cmd.is_url(protocol_only_url):
        tmp = protocol_only_url.rstrip("/")
        filename = tmp.split("/")[-1]

        assert ":" not in filename, \
            f"is_url({protocol_only_url}) returns True but produces invalid filename: {filename}"

        assert len(filename) > 0, \
            f"is_url({protocol_only_url}) returns True but produces empty filename"
```

**Failing input**: `"http://"` (also fails for `"https://"` and `"ftp://"`)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.management.templates import TemplateCommand

cmd = TemplateCommand()

url = "http://"
print(f"Testing URL: {url}")
print(f"is_url() returns: {cmd.is_url(url)}")

tmp = url.rstrip("/")
filename = tmp.split("/")[-1]
print(f"Extracted filename: '{filename}'")
print(f"Filename contains colon: {':' in filename}")
```

**Output:**
```
Testing URL: http://
is_url() returns: True
Extracted filename: 'http:'
Filename contains colon: True
```

## Why This Is A Bug

1. **Invalid Filenames**: The extracted filename `"http:"` contains a colon, which is invalid on Windows systems (colons are reserved for drive letters).

2. **Non-Downloadable URLs**: Protocol-only URLs like `"http://"` cannot actually be downloaded - they lack a hostname. If a user attempts to use such a URL with `django-admin startproject --template http://`, the download will fail after `is_url()` incorrectly validates it.

3. **Violates Method Contract**: The `is_url()` method's docstring states it "Return[s] True if the name looks like a URL." A protocol without a hostname is not a valid URL according to RFC 3986, which requires an authority component after `://`.

## Fix

The `is_url()` method should validate that the URL contains more than just the protocol. A simple fix is to check that there's content after the protocol separator:

```diff
diff --git a/django/core/management/templates.py b/django/core/management/templates.py
index 1234567..abcdefg 100644
--- a/django/core/management/templates.py
+++ b/django/core/management/templates.py
@@ -381,6 +381,10 @@ class TemplateCommand(BaseCommand):
     def is_url(self, template):
         """Return True if the name looks like a URL."""
         if ":" not in template:
             return False
         scheme = template.split(":", 1)[0].lower()
-        return scheme in self.url_schemes
+        if scheme not in self.url_schemes:
+            return False
+        # Ensure there's content after the scheme (not just "http://")
+        remainder = template.split(":", 1)[1]
+        return len(remainder.lstrip("/")) > 0
```

This ensures that URLs like `"http://"` are rejected, while valid URLs like `"http://example.com"` continue to pass.