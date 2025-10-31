# Bug Report: django.core.handlers.wsgi.LimitedStream Negative Content-Length

**Target**: `django.core.handlers.wsgi.LimitedStream`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`LimitedStream` fails to enforce read limits when initialized with a negative limit value, allowing unlimited reading from the underlying stream. This can occur when a malicious client sends a negative `Content-Length` header.

## Property-Based Test

```python
from io import BytesIO
from hypothesis import given, strategies as st, settings
from django.core.handlers.wsgi import LimitedStream

@given(
    data=st.binary(min_size=1, max_size=1000),
    limit=st.integers(min_value=-1000, max_value=-1)
)
@settings(max_examples=100)
def test_limitedstream_negative_limit_blocks_reading(data, limit):
    stream = BytesIO(data)
    limited = LimitedStream(stream, limit)
    result = limited.read()
    assert len(result) == 0, \
        f"With negative limit {limit}, read {len(result)} bytes from {len(data)} byte stream"
```

**Failing input**: `data=b'X' * 100, limit=-10`

## Reproducing the Bug

```python
from io import BytesIO
from django.core.handlers.wsgi import LimitedStream

data = b'A' * 1000
negative_limit = -50

stream = BytesIO(data)
limited = LimitedStream(stream, negative_limit)
result = limited.read()

print(f"Limit: {negative_limit}")
print(f"Bytes read: {len(result)}")
```

Output:
```
Limit: -50
Bytes read: 1000
```

## Why This Is A Bug

The `LimitedStream` class is designed to "disallow reading it past a number of bytes" (wsgi.py:17). When initialized with a negative limit:

1. The check `if _pos >= limit` (line 32) evaluates to `False` when `_pos=0` and `limit=-50`
2. The size calculation `size = limit - _pos` results in a negative number (e.g., `-50`)
3. Python's `stream.read(negative_number)` reads **all** remaining data
4. This completely bypasses the intended limit enforcement

This can be triggered in `WSGIRequest.__init__` (wsgi.py:75-78):
```python
try:
    content_length = int(environ.get("CONTENT_LENGTH"))
except (ValueError, TypeError):
    content_length = 0
self._stream = LimitedStream(self.environ["wsgi.input"], content_length)
```

A malicious client sending `Content-Length: -1000` would successfully parse as an integer and create a `LimitedStream` with a negative limit, allowing unbounded reads.

## Fix

```diff
--- a/django/core/handlers/wsgi.py
+++ b/django/core/handlers/wsgi.py
@@ -74,7 +74,7 @@ class WSGIRequest(HttpRequest):
         try:
             content_length = int(environ.get("CONTENT_LENGTH"))
         except (ValueError, TypeError):
             content_length = 0
-        self._stream = LimitedStream(self.environ["wsgi.input"], content_length)
+        self._stream = LimitedStream(self.environ["wsgi.input"], max(0, content_length))
         self._read_started = False
         self.resolver_match = None
```

Alternative fix in `LimitedStream.__init__`:
```diff
--- a/django/core/handlers/wsgi.py
+++ b/django/core/handlers/wsgi.py
@@ -23,7 +23,7 @@ class LimitedStream(IOBase):
     def __init__(self, stream, limit):
         self._read = stream.read
         self._readline = stream.readline
         self._pos = 0
-        self.limit = limit
+        self.limit = max(0, limit)
```