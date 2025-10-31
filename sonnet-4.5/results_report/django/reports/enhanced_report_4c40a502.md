# Bug Report: django.core.handlers.wsgi.get_script_name UnicodeDecodeError on Invalid UTF-8

**Target**: `django.core.handlers.wsgi.get_script_name`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_script_name` function crashes with `UnicodeDecodeError` when processing WSGI environ values containing invalid UTF-8 sequences, while similar functions in the same module handle such input gracefully using error recovery mechanisms.

## Property-Based Test

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
from hypothesis import given, strategies as st, example

@given(
    parts=st.lists(st.binary(min_size=0, max_size=20), min_size=2, max_size=5),
    path_info=st.binary(min_size=0, max_size=50)
)
@example(parts=[b'', b'\x80'], path_info=b'')  # The specific failing case
def test_get_script_name_with_multiple_slashes(parts, path_info):
    script_url = b'//'.join(parts)

    environ = {
        'SCRIPT_URL': script_url.decode('latin1', errors='replace'),
        'PATH_INFO': path_info.decode('latin1', errors='replace'),
        'SCRIPT_NAME': ''
    }

    try:
        script_name = get_script_name(environ)
        assert '//' not in script_name
        print(f"✓ Test passed for parts={parts}, path_info={path_info}")
    except UnicodeDecodeError as e:
        print(f"✗ UnicodeDecodeError with parts={parts}, path_info={path_info}")
        print(f"  Error: {e}")
        raise

# Run the test
if __name__ == "__main__":
    print("Running Hypothesis test with specific failing example...")
    test_get_script_name_with_multiple_slashes()
```

<details>

<summary>
**Failing input**: `parts=[b'', b'\x80'], path_info=b''`
</summary>
```
Running Hypothesis test with specific failing example...
✗ UnicodeDecodeError with parts=[b'', b'\x80'], path_info=b''
  Error: 'utf-8' codec can't decode byte 0x80 in position 1: invalid start byte
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 40, in <module>
    test_get_script_name_with_multiple_slashes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 15, in test_get_script_name_with_multiple_slashes
    parts=st.lists(st.binary(min_size=0, max_size=20), min_size=2, max_size=5),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 29, in test_get_script_name_with_multiple_slashes
    script_name = get_script_name(environ)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/handlers/wsgi.py", line 184, in get_script_name
    return script_name.decode()
           ~~~~~~~~~~~~~~~~~~^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 1: invalid start byte
Falsifying explicit example: test_get_script_name_with_multiple_slashes(
    parts=[b'', b'\x80'],
    path_info=b'',
)
```
</details>

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

try:
    result = get_script_name(environ)
    print(f"Success: get_script_name returned: {repr(result)}")
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}")
    print(f"  Error at position {e.start}: byte {hex(e.object[e.start])}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
UnicodeDecodeError when calling get_script_name with invalid UTF-8
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/63/repo.py", line 20, in <module>
    result = get_script_name(environ)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/handlers/wsgi.py", line 184, in get_script_name
    return script_name.decode()
           ~~~~~~~~~~~~~~~~~~^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
  Error at position 0: byte 0x80
```
</details>

## Why This Is A Bug

This is a legitimate bug for three key reasons:

1. **Inconsistent Error Handling Within Same Module**: The `django.core.handlers.wsgi` module contains three similar functions that process WSGI environ data, but they handle invalid UTF-8 differently:
   - `get_path_info()` (line 151): Uses `repercent_broken_unicode(path_info).decode()` which converts invalid UTF-8 bytes to percent-encoded strings (e.g., '\x80' becomes '%80')
   - `get_str_from_wsgi()` (line 207): Uses `value.decode(errors="replace")` which replaces invalid sequences with the Unicode replacement character '�'
   - `get_script_name()` (line 184): Uses bare `script_name.decode()` without any error handling, causing it to crash

2. **Violates WSGI Specification Robustness**: According to PEP 3333, WSGI environ values must be decoded using ISO-8859-1 (Latin-1), which can represent any byte value from 0x00 to 0xFF. When Django re-encodes these strings back to bytes using `get_bytes_from_wsgi()` and then attempts to decode as UTF-8, it must handle cases where the original bytes were not valid UTF-8. The byte 0x80 is valid in ISO-8859-1 but invalid as a UTF-8 start byte.

3. **Creates Security Vulnerability**: This crash can be triggered by external user input (the URL), creating a denial-of-service vector. Malicious actors can send HTTP requests with URLs containing invalid UTF-8 sequences to crash Django applications. While some web servers might filter such requests, Django should not assume this protection exists.

## Relevant Context

The WSGI specification (PEP 3333) mandates that all strings in the environ dictionary be encoded using ISO-8859-1. This encoding was chosen specifically because it can decode any byte sequence without errors. Django's strategy is to:
1. Receive ISO-8859-1 decoded strings from the WSGI server
2. Re-encode them back to bytes using ISO-8859-1 (recovering original bytes)
3. Decode those bytes as UTF-8 for internal processing

The `repercent_broken_unicode()` function in `django/utils/encoding.py` was specifically created to handle this situation safely. It attempts UTF-8 decoding and percent-encodes any invalid byte sequences to prevent crashes (addressing CVE-2019-14235).

Testing confirms the inconsistency:
- `get_path_info('\x80')` returns `'%80'` (safe)
- `get_str_from_wsgi(environ, 'KEY', '')` with KEY='\x80' returns `'�'` (safe)
- `get_script_name()` with SCRIPT_URL='\x80' crashes with UnicodeDecodeError

Documentation: https://docs.djangoproject.com/en/stable/ref/request-response/
Source code: https://github.com/django/django/blob/main/django/core/handlers/wsgi.py

## Proposed Fix

```diff
--- a/django/core/handlers/wsgi.py
+++ b/django/core/handlers/wsgi.py
@@ -181,7 +181,7 @@ def get_script_name(environ):
     else:
         script_name = get_bytes_from_wsgi(environ, "SCRIPT_NAME", "")

-    return script_name.decode()
+    return repercent_broken_unicode(script_name).decode()
```

This fix makes `get_script_name` consistent with `get_path_info` by using the same `repercent_broken_unicode` function to safely handle invalid UTF-8 sequences.