# Bug Report: Django Core ServerHandler Negative Content-Length Acceptance

**Target**: `django.core.servers.basehttp.ServerHandler.__init__`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

ServerHandler incorrectly accepts negative Content-Length values, creating a LimitedStream with a negative limit instead of normalizing to 0, violating HTTP RFC 7230 specifications.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.servers.basehttp import ServerHandler
from io import BytesIO

@given(st.integers())
def test_serverhandler_content_length_parsing_integers(content_length):
    stdin = BytesIO(b"test data")
    stdout = BytesIO()
    stderr = BytesIO()

    environ = {"CONTENT_LENGTH": str(content_length), "REQUEST_METHOD": "POST"}

    handler = ServerHandler(stdin, stdout, stderr, environ)

    expected_limit = max(0, content_length)
    assert handler.stdin.limit == expected_limit, f"Expected {expected_limit}, got {handler.stdin.limit}"

# Run the test
if __name__ == "__main__":
    test_serverhandler_content_length_parsing_integers()
```

<details>

<summary>
**Failing input**: `content_length=-1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 20, in <module>
    test_serverhandler_content_length_parsing_integers()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 6, in test_serverhandler_content_length_parsing_integers
    def test_serverhandler_content_length_parsing_integers(content_length):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 16, in test_serverhandler_content_length_parsing_integers
    assert handler.stdin.limit == expected_limit, f"Expected {expected_limit}, got {handler.stdin.limit}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 0, got -1
Falsifying example: test_serverhandler_content_length_parsing_integers(
    content_length=-1,
)
```
</details>

## Reproducing the Bug

```python
from django.core.servers.basehttp import ServerHandler
from io import BytesIO

# Test case demonstrating the bug with negative Content-Length
stdin = BytesIO(b"request body data")
stdout = BytesIO()
stderr = BytesIO()
environ = {"CONTENT_LENGTH": "-1", "REQUEST_METHOD": "POST"}

handler = ServerHandler(stdin, stdout, stderr, environ)

print(f"LimitedStream limit: {handler.stdin.limit}")
print(f"Expected limit: 0 (should normalize negative to 0)")
print(f"Actual limit: {handler.stdin.limit}")

# Verify the bug exists
assert handler.stdin.limit == -1, f"Bug confirmed: negative limit {handler.stdin.limit} instead of 0"
print("\nBug confirmed: ServerHandler accepts negative Content-Length values")
print("This violates HTTP RFC 7230 which requires non-negative Content-Length")
```

<details>

<summary>
ServerHandler creates LimitedStream with negative limit
</summary>
```
LimitedStream limit: -1
Expected limit: 0 (should normalize negative to 0)
Actual limit: -1

Bug confirmed: ServerHandler accepts negative Content-Length values
This violates HTTP RFC 7230 which requires non-negative Content-Length
```
</details>

## Why This Is A Bug

This violates HTTP RFC 7230 Section 3.3.2, which explicitly states that "Any Content-Length field value greater than or equal to zero is valid." The specification only permits non-negative decimal numbers for Content-Length headers.

The ServerHandler code at lines 124-127 attempts to handle invalid Content-Length values by catching ValueError (for non-numeric strings) and TypeError (for None values), defaulting them to 0. However, it fails to validate that successfully parsed integers are non-negative. This creates an inconsistent error handling pattern where:

- Invalid strings like "abc" → 0 (correct)
- None/missing Content-Length → 0 (correct)
- Negative integers like "-1" → -1 (incorrect, should be 0)

When a LimitedStream is created with a negative limit, all read operations return empty bytes because the condition `self._pos >= self.limit` in lines 32 and 45 of LimitedStream immediately evaluates to True (0 >= -1). This causes request bodies to appear empty, creating silent failures that are difficult to debug.

## Relevant Context

The same vulnerable pattern exists in WSGIRequest at `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/handlers/wsgi.py:74-77`, indicating this is a systematic issue in Django's WSGI handling code.

The LimitedStream implementation is based on werkzeug.wsgi.LimitedStream (as noted in the docstring at lines 16-21), which also expects non-negative limits. The Django development server (runserver) uses this ServerHandler for local development, though production deployments typically use other WSGI servers.

Related code locations:
- ServerHandler.__init__: `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/servers/basehttp.py:117-130`
- WSGIRequest.__init__: `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/handlers/wsgi.py:57-80`
- LimitedStream: `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/handlers/wsgi.py:15-54`

## Proposed Fix

```diff
--- a/django/core/servers/basehttp.py
+++ b/django/core/servers/basehttp.py
@@ -124,6 +124,7 @@ class ServerHandler(simple_server.ServerHandler):
         try:
             content_length = int(environ.get("CONTENT_LENGTH"))
         except (ValueError, TypeError):
             content_length = 0
+        content_length = max(0, content_length)
         super().__init__(
             LimitedStream(stdin, content_length), stdout, stderr, environ, **kwargs
         )
```