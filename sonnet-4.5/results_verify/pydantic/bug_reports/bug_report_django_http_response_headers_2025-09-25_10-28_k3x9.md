# Bug Report: django.http.HttpResponse Header Value Encoding Asymmetry

**Target**: `django.http.HttpResponse` (specifically `ResponseHeaders.__setitem__` and `__getitem__`)
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Setting a header with a non-latin-1 Unicode value and then retrieving it returns the MIME-encoded form instead of the original value, violating the basic contract that `response[header] == value` after `response[header] = value`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from django.http import HttpResponse

@given(st.text(min_size=1, max_size=50),
       st.text(max_size=100))
def test_httpresponse_header_roundtrip(header_name, header_value):
    assume(header_name.strip() != '')
    assume('\n' not in header_name and '\r' not in header_name)
    assume('\n' not in header_value and '\r' not in header_value)

    response = HttpResponse()
    response[header_name] = header_value

    assert response[header_name] == header_value
```

**Failing input**: `header_name='X-Test', header_value='Ā'` (or any non-latin-1 character)

## Reproducing the Bug

```python
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

from django.conf import settings
if not settings.configured:
    settings.configure(SECRET_KEY='test-secret-key', DEFAULT_CHARSET='utf-8')

from django.http import HttpResponse

response = HttpResponse()
original_value = 'Ā'
response['X-Custom-Header'] = original_value

retrieved_value = response['X-Custom-Header']

print(f"Set value: {original_value!r}")
print(f"Got value: {retrieved_value!r}")
print(f"Equal: {original_value == retrieved_value}")
```

Output:
```
Set value: 'Ā'
Got value: '=?utf-8?b?xIA=?='
Equal: False
```

## Why This Is A Bug

The `ResponseHeaders.__setitem__` method calls `_convert_to_charset(value, "latin-1", mime_encode=True)` which MIME-encodes values that can't be represented in latin-1. However, `__getitem__` simply returns the stored value without decoding it back.

This violates the basic property that setting and then getting a value should return the original value. Users expect `response[key]` to return what they set, not the wire-format encoding. The encoding/decoding should be symmetric - if Django encodes on set, it should decode on get.

This creates confusion and bugs in user code that sets custom headers and expects to retrieve the original values.

## Fix

The fix should add MIME-decoding in the `__getitem__` method to match the encoding done in `__setitem__`. Here's a patch:

```diff
--- a/django/http/response.py
+++ b/django/http/response.py
@@ -1,6 +1,7 @@
 import io
 import sys
 from email.header import Header
+from email import message_from_string
 from http.client import responses
 from urllib.parse import quote, urlparse

@@ -92,7 +93,14 @@ class ResponseHeaders(CaseInsensitiveMa

     def __getitem__(self, key):
-        return self._store[key.lower()][1]
+        value = self._store[key.lower()][1]
+        # Decode MIME-encoded values (RFC 2047) back to Unicode
+        if value.startswith('=?') and value.endswith('?='):
+            try:
+                # Parse the MIME-encoded header
+                from email.header import decode_header
+                decoded_parts = decode_header(value)
+                return ''.join(text.decode(charset or 'utf-8') if isinstance(text, bytes) else text
+                             for text, charset in decoded_parts)
+            except Exception:
+                pass  # If decoding fails, return the original value
+        return value
```