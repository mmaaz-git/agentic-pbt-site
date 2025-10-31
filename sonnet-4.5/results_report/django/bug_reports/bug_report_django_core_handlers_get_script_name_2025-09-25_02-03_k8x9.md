# Bug Report: django.core.handlers.wsgi.get_script_name UnicodeDecodeError

**Target**: `django.core.handlers.wsgi.get_script_name`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_script_name` function crashes with `UnicodeDecodeError` when processing WSGI environ values containing invalid UTF-8 sequences, while similar functions (`get_path_info`, `get_str_from_wsgi`) handle such input gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(
    parts=st.lists(st.binary(min_size=0, max_size=20), min_size=2, max_size=5),
    path_info=st.binary(min_size=0, max_size=50)
)
def test_get_script_name_with_multiple_slashes(parts, path_info):
    script_url = b'//'.join(parts)

    environ = {
        'SCRIPT_URL': script_url.decode('latin1', errors='replace'),
        'PATH_INFO': path_info.decode('latin1', errors='replace'),
        'SCRIPT_NAME': ''
    }

    script_name = get_script_name(environ)
    assert '//' not in script_name
```

**Failing input**: `parts=[b'', b'\x80'], path_info=b''`

## Reproducing the Bug

```python
import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        FORCE_SCRIPT_NAME=None,
    )
    django.setup()

from django.core.handlers.wsgi import get_script_name

environ = {
    'SCRIPT_URL': '\x80',
    'PATH_INFO': '',
    'SCRIPT_NAME': ''
}

get_script_name(environ)
```

## Why This Is A Bug

1. **Inconsistent error handling**: Similar functions handle invalid UTF-8 gracefully:
   - `get_path_info()` uses `repercent_broken_unicode(path_info).decode()` (wsgi.py:151)
   - `get_str_from_wsgi()` uses `value.decode(errors="replace")` (wsgi.py:207)
   - `get_script_name()` uses bare `script_name.decode()` (wsgi.py:184)

2. **Realistic scenario**: WSGI servers can receive arbitrary bytes. The WSGI spec requires environ values to be decoded using ISO-8859-1 (which accepts all byte values). When Django re-encodes to recover original bytes and those bytes aren't valid UTF-8, the crash occurs.

3. **Impact**: Malicious or misconfigured clients can crash Django servers by sending URLs with invalid UTF-8 sequences.

## Fix

```diff
diff --git a/django/core/handlers/wsgi.py b/django/core/handlers/wsgi.py
index 1234567..abcdefg 100644
--- a/django/core/handlers/wsgi.py
+++ b/django/core/handlers/wsgi.py
@@ -181,7 +181,7 @@ def get_script_name(environ):
     else:
         script_name = get_bytes_from_wsgi(environ, "SCRIPT_NAME", "")

-    return script_name.decode()
+    return script_name.decode(errors="replace")
```

This fix makes `get_script_name` consistent with `get_str_from_wsgi` and prevents crashes on invalid UTF-8 input.