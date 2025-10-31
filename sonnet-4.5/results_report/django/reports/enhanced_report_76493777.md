# Bug Report: Django TemplateCommand is_url Accepts Invalid Protocol-Only URLs

**Target**: `django.core.management.templates.TemplateCommand.is_url`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_url()` method incorrectly validates protocol-only URLs like `"http://"`, `"https://"`, and `"ftp://"` as valid URLs, which leads to invalid Windows filenames containing colons and represents non-downloadable URLs.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

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
            f"is_url({protocol_only_url!r}) returns True but produces invalid filename: {filename!r}"

        assert len(filename) > 0, \
            f"is_url({protocol_only_url!r}) returns True but produces empty filename"

# Run the test
if __name__ == "__main__":
    test_is_url_rejects_protocol_only_urls()
```

<details>

<summary>
**Failing input**: `'http://'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 25, in <module>
    test_is_url_rejects_protocol_only_urls()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 8, in test_is_url_rejects_protocol_only_urls
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 17, in test_is_url_rejects_protocol_only_urls
    assert ":" not in filename, \
           ^^^^^^^^^^^^^^^^^^^
AssertionError: is_url('http://') returns True but produces invalid filename: 'http:'
Falsifying example: test_is_url_rejects_protocol_only_urls(
    protocol='http',
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.management.templates import TemplateCommand

cmd = TemplateCommand()

# Test protocol-only URLs
test_urls = ["http://", "https://", "ftp://"]

for url in test_urls:
    print(f"\nTesting URL: {url!r}")
    print(f"is_url() returns: {cmd.is_url(url)}")

    # Show what filename would be extracted (from download method's cleanup_url logic)
    tmp = url.rstrip("/")
    filename = tmp.split("/")[-1]
    print(f"Extracted filename: {filename!r}")
    print(f"Filename contains colon: {':' in filename}")
    print(f"Filename length: {len(filename)}")

    # Check if this is valid on Windows
    if ':' in filename:
        print(f"ERROR: Filename {filename!r} is invalid on Windows (contains colon)")
```

<details>

<summary>
Invalid Windows filenames generated from protocol-only URLs
</summary>
```

Testing URL: 'http://'
is_url() returns: True
Extracted filename: 'http:'
Filename contains colon: True
Filename length: 5
ERROR: Filename 'http:' is invalid on Windows (contains colon)

Testing URL: 'https://'
is_url() returns: True
Extracted filename: 'https:'
Filename contains colon: True
Filename length: 6
ERROR: Filename 'https:' is invalid on Windows (contains colon)

Testing URL: 'ftp://'
is_url() returns: True
Extracted filename: 'ftp:'
Filename contains colon: True
Filename length: 4
ERROR: Filename 'ftp:' is invalid on Windows (contains colon)
```
</details>

## Why This Is A Bug

The `is_url()` method incorrectly validates protocol-only URLs, which causes multiple issues:

1. **Windows File System Incompatibility**: The `download()` method uses `cleanup_url()` to extract filenames from URLs (lines 296-303 in templates.py). For protocol-only URLs like `"http://"`, this produces filenames like `"http:"` which are invalid on Windows. Windows reserves colons for drive letters (e.g., `C:`), making these filenames unusable.

2. **RFC 3986 Violation**: According to RFC 3986, a valid URI with the http/https/ftp schemes requires an authority component (hostname) after the `://`. The string `"http://"` lacks this required component and is not a valid URI.

3. **Non-Downloadable URLs**: The purpose of `is_url()` is to identify URLs that can be downloaded as templates (used in line 248-250 of templates.py). Protocol-only URLs cannot be downloaded because they lack a hostname/server to connect to. If passed to `download()`, the operation would fail with a network error.

4. **Method Contract Violation**: While the docstring states the method returns True if "the name looks like a URL", the method's usage context in `handle_template()` clearly expects it to validate downloadable template sources, not just any string with a protocol prefix.

## Relevant Context

The `is_url()` method is part of Django's management command infrastructure, specifically used by `startproject` and `startapp` commands to support downloading templates from URLs via the `--template` option.

The current implementation (lines 381-386 of `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/management/templates.py`):
- Only checks if the string contains a colon
- Verifies the scheme is in `["http", "https", "ftp"]`
- Does not validate that there's actual content after the protocol

The Django documentation for the `--template` option states it accepts "The path or URL to load the template from", implying valid, functional URLs should be provided.

## Proposed Fix

```diff
--- a/django/core/management/templates.py
+++ b/django/core/management/templates.py
@@ -383,4 +383,9 @@ class TemplateCommand(BaseCommand):
         if ":" not in template:
             return False
         scheme = template.split(":", 1)[0].lower()
-        return scheme in self.url_schemes
+        if scheme not in self.url_schemes:
+            return False
+        # Ensure there's a hostname after the scheme
+        # (not just "http://", "https://", or "ftp://")
+        remainder = template[len(scheme) + 3:]  # Skip "scheme://"
+        return bool(remainder)
```