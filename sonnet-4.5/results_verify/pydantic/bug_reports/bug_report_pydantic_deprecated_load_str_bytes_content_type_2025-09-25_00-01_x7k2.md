# Bug Report: pydantic.deprecated.parse.load_str_bytes Rejects Valid Content-Type Headers with Parameters

**Target**: `pydantic.deprecated.parse.load_str_bytes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `load_str_bytes` function incorrectly rejects valid HTTP Content-Type headers that include parameters (e.g., `application/json; charset=utf-8`). The function uses `str.endswith()` to check content types, which fails when the content type includes standard parameters like charset.

## Property-Based Test

```python
import json
from hypothesis import given, strategies as st, example
from pydantic.deprecated.parse import load_str_bytes


@given(st.sampled_from([
    'application/json; charset=utf-8',
    'application/json;charset=utf-8',
    'text/json; encoding=utf-8',
]))
@example('application/json; charset=utf-8')
def test_load_str_bytes_handles_content_type_with_parameters(content_type):
    test_data = {"key": "value"}
    json_str = json.dumps(test_data)

    result = load_str_bytes(json_str, content_type=content_type)
    assert result == test_data
```

**Failing input**: `content_type='application/json; charset=utf-8'`

## Reproducing the Bug

```python
import json
from pydantic.deprecated.parse import load_str_bytes

test_data = {"key": "value"}
json_str = json.dumps(test_data)

try:
    result = load_str_bytes(json_str, content_type='application/json; charset=utf-8')
    print(f"Success: {result}")
except TypeError as e:
    print(f"Error: {e}")
```

Output:
```
Error: Unknown content-type: application/json; charset=utf-8
```

## Why This Is A Bug

HTTP Content-Type headers commonly include parameters according to RFC 2045 and RFC 7231. The most common parameter is `charset`, which specifies the character encoding. For example:
- `application/json; charset=utf-8`
- `text/javascript; charset=ISO-8859-1`

The current implementation uses `content_type.endswith(('json', 'javascript'))` to detect JSON content. This check fails when the content type includes parameters because:
- `'application/json; charset=utf-8'.endswith('json')` → `False` (ends with '8')

This causes the function to incorrectly raise `TypeError: Unknown content-type` for valid, standards-compliant Content-Type headers.

## Fix

```diff
--- a/pydantic/deprecated/parse.py
+++ b/pydantic/deprecated/parse.py
@@ -34,9 +34,14 @@ def load_str_bytes(
 ) -> Any:
     warnings.warn('`load_str_bytes` is deprecated.', category=PydanticDeprecatedSince20, stacklevel=2)
     if proto is None and content_type:
-        if content_type.endswith(('json', 'javascript')):
+        content_type_base = content_type.split(';')[0].strip()
+
+        if 'json' in content_type_base or 'javascript' in content_type_base:
             pass
-        elif allow_pickle and content_type.endswith('pickle'):
+        elif allow_pickle and 'pickle' in content_type_base:
             proto = Protocol.pickle
         else:
             raise TypeError(f'Unknown content-type: {content_type}')
```

Alternative fix using more robust content-type parsing:
```diff
--- a/pydantic/deprecated/parse.py
+++ b/pydantic/deprecated/parse.py
@@ -34,9 +34,11 @@ def load_str_bytes(
 ) -> Any:
     warnings.warn('`load_str_bytes` is deprecated.', category=PydanticDeprecatedSince20, stacklevel=2)
     if proto is None and content_type:
-        if content_type.endswith(('json', 'javascript')):
+        content_type_lower = content_type.lower()
+        if 'json' in content_type_lower or 'javascript' in content_type_lower:
             pass
-        elif allow_pickle and content_type.endswith('pickle'):
+        elif allow_pickle and 'pickle' in content_type_lower:
             proto = Protocol.pickle
         else:
             raise TypeError(f'Unknown content-type: {content_type}')