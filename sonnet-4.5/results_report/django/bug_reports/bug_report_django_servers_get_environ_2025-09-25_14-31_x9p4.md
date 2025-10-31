# Bug Report: django.core.servers.basehttp.WSGIRequestHandler.get_environ Dictionary Modification During Iteration

**Target**: `django.core.servers.basehttp.WSGIRequestHandler.get_environ`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`WSGIRequestHandler.get_environ()` modifies `self.headers` dictionary while iterating over it, which can cause `RuntimeError` or result in incomplete header removal, leaving security-sensitive headers with underscores intact.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.servers.basehttp import WSGIRequestHandler
from email.message import Message
from unittest.mock import Mock

@given(
    st.lists(
        st.tuples(
            st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1),
            st.text()
        ),
        min_size=1,
        max_size=20
    )
)
def test_get_environ_removes_all_underscore_headers(headers_list):
    mock_handler = Mock(spec=WSGIRequestHandler)

    headers = Message()
    for key, value in headers_list:
        headers[key] = value

    mock_handler.headers = headers
    mock_handler.get_stderr = Mock(return_value=Mock())

    original_keys_with_underscores = [k for k in headers.keys() if '_' in k]

    result = WSGIRequestHandler.get_environ(mock_handler)

    remaining_keys_with_underscores = [k for k in mock_handler.headers.keys() if '_' in k]
    assert remaining_keys_with_underscores == [], \
        f"Headers with underscores not fully removed: {remaining_keys_with_underscores}"
```

**Failing input**: Any request with multiple headers containing underscores, e.g.:
```python
headers = {
    'X_Custom_Header': 'value1',
    'Another_Header': 'value2',
    'Third_Header': 'value3'
}
```

## Reproducing the Bug

```python
import sys
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')
django.setup()

from django.core.servers.basehttp import WSGIRequestHandler
from email.message import Message
from unittest.mock import Mock

mock_handler = Mock(spec=WSGIRequestHandler)

headers = Message()
headers['X_Header_One'] = 'value1'
headers['Y_Header_Two'] = 'value2'
headers['Z_Header_Three'] = 'value3'
headers['Normal-Header'] = 'value4'

mock_handler.headers = headers
mock_handler.get_stderr = Mock(return_value=Mock())

print("Headers before:", list(headers.keys()))

try:
    WSGIRequestHandler.get_environ(mock_handler)
except RuntimeError as e:
    print(f"RuntimeError: {e}")

print("Headers after:", list(headers.keys()))
print("Remaining underscore headers:", [k for k in headers.keys() if '_' in k])
```

**Expected**: All headers with underscores removed
**Actual**: Either `RuntimeError: dictionary changed size during iteration` or incomplete removal

## Why This Is A Bug

The security comment explicitly states the purpose is to "Strip all headers with underscores" to prevent header-spoofing. However, the implementation violates Python's rule against modifying a dictionary during iteration:

1. **Crash risk**: Can raise `RuntimeError` when the underlying dictionary implementation detects modification during iteration
2. **Security risk**: Incomplete removal means some underscore-containing headers may survive, defeating the security measure
3. **Undefined behavior**: Which headers get removed depends on dictionary iteration order

This is particularly serious because it's a security feature - incomplete header removal creates the exact vulnerability the code is trying to prevent.

## Fix

Collect keys to delete first, then delete them:

```diff
     def get_environ(self):
         # Strip all headers with underscores in the name before constructing
         # the WSGI environ. This prevents header-spoofing based on ambiguity
         # between underscores and dashes both normalized to underscores in WSGI
         # env vars. Nginx and Apache 2.4+ both do this as well.
-        for k in self.headers:
-            if "_" in k:
-                del self.headers[k]
+        keys_to_delete = [k for k in self.headers if "_" in k]
+        for k in keys_to_delete:
+            del self.headers[k]

         return super().get_environ()
```