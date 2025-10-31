# Bug Report: django.core.mail.forbid_multi_line_headers Header Injection Vulnerability

**Target**: `django.core.mail.message.forbid_multi_line_headers`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `forbid_multi_line_headers` function fails to prevent newlines in email headers when the header value contains non-ASCII characters. This creates a **header injection vulnerability** that the function is explicitly designed to prevent.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.mail.message import forbid_multi_line_headers

@given(st.text(), st.text(min_size=1), st.sampled_from(['utf-8', 'ascii', 'iso-8859-1', None]))
def test_forbid_multi_line_headers_rejects_newlines(name, val, encoding):
    """
    Property: forbid_multi_line_headers should never return a value containing newlines
    """
    if '\n' in val or '\r' in val:
        with pytest.raises(BadHeaderError):
            forbid_multi_line_headers(name, val, encoding)
    else:
        result_name, result_val = forbid_multi_line_headers(name, val, encoding)
        assert '\n' not in result_val
        assert '\r' not in result_val
```

**Failing input**: `forbid_multi_line_headers('X-Custom-Header', '0\x0c\x80', 'utf-8')`

## Reproducing the Bug

```python
from django.core.mail.message import forbid_multi_line_headers

name = 'X-Custom-Header'
val = '0\x0c\x80'
encoding = 'utf-8'

result_name, result_val = forbid_multi_line_headers(name, val, encoding)

print(f"Input: {repr(val)}")
print(f"Output: {repr(result_val)}")
print(f"Contains newline: {'\\n' in result_val}")
```

**Output:**
```
Input: '0\x0c\x80'
Output: '=?utf-8?q?0?=\n =?utf-8?b?IMKA?='
Contains newline: True
```

## Why This Is A Bug

The function `forbid_multi_line_headers` has a single documented purpose: "Forbid multi-line headers to prevent header injection." However, when the header value contains non-ASCII characters, the function encodes it using Python's `email.header.Header.encode()` method (line 72 in message.py). This method can introduce newlines as part of RFC 2047 encoding to keep lines under the maximum length.

The result is that an attacker could craft header values with non-ASCII characters that, after encoding, contain newlines. This defeats the security protection and allows header injection attacks, which could enable:
- Email header spoofing
- Injection of additional headers
- Potential email body injection

The function checks for newlines in the *input* (line 60) but fails to validate that its own encoding process doesn't introduce newlines in the *output*.

## Fix

The fix should ensure that after encoding, the result is checked for newlines, or the encoding should use parameters that prevent line breaks. Here's a patch:

```diff
--- a/django/core/mail/message.py
+++ b/django/core/mail/message.py
@@ -69,8 +69,12 @@ def forbid_multi_line_headers(name, val, encoding):
                 sanitize_address(addr, encoding) for addr in getaddresses((val,))
             )
         else:
-            val = Header(val, encoding).encode()
+            val = Header(val, encoding).encode(splitchars=' ')
     else:
         if name.lower() == "subject":
-            val = Header(val).encode()
+            val = Header(val).encode(splitchars=' ')
+
+    if "\n" in val or "\r" in val:
+        raise BadHeaderError(
+            "Header encoding introduced newlines (got %r for header %r)" % (val, name)
+        )
     return name, val
```

The `splitchars=' '` parameter tells the Header encoder to only split on spaces, which is safer. Additionally, a final check after encoding ensures no newlines were introduced through any code path.