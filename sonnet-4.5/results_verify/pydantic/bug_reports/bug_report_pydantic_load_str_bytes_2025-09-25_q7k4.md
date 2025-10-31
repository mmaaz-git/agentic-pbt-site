# Bug Report: pydantic.deprecated.parse.load_str_bytes Rejects Content-Type with Parameters

**Target**: `pydantic.deprecated.parse.load_str_bytes`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `load_str_bytes` function incorrectly rejects valid HTTP Content-Type headers that include parameters such as charset. For example, `'application/json; charset=utf-8'` is rejected as an "Unknown content-type" even though it's a standard JSON Content-Type header.

## Property-Based Test

```python
import json
from hypothesis import given, strategies as st
from pydantic.deprecated.parse import load_str_bytes


@given(st.dictionaries(st.text(min_size=1), st.integers()))
def test_load_str_bytes_with_charset_parameter(data_dict):
    json_str = json.dumps(data_dict)

    result1 = load_str_bytes(json_str, content_type='application/json')
    assert result1 == data_dict

    result2 = load_str_bytes(json_str, content_type='application/json; charset=utf-8')
    assert result2 == data_dict, "Content-Type with charset parameter should work"
```

**Failing input**: `data_dict={}` (or any dict), `content_type='application/json; charset=utf-8'`

## Reproducing the Bug

```python
from pydantic.deprecated.parse import load_str_bytes
import json

data = '{"key": "value"}'

result1 = load_str_bytes(data, content_type='application/json')
print(f"Without charset: {result1}")

try:
    result2 = load_str_bytes(data, content_type='application/json; charset=utf-8')
    print(f"With charset: {result2}")
except TypeError as e:
    print(f"TypeError: {e}")
```

Output:
```
Without charset: {'key': 'value'}
TypeError: Unknown content-type: application/json; charset=utf-8
```

## Why This Is A Bug

HTTP Content-Type headers commonly include parameters, especially the `charset` parameter. According to RFC 2616 and RFC 7231, a Content-Type header like `'application/json; charset=utf-8'` is not only valid but recommended for specifying character encoding.

The current implementation uses `content_type.endswith(('json', 'javascript'))` to detect JSON content types (line 37), which fails when the Content-Type includes parameters because the string no longer ends with 'json' or 'javascript' - it ends with the parameter value (e.g., 'utf-8').

This breaks real-world usage where HTTP responses include charset parameters, which is standard practice for web APIs.

## Fix

The function should check if the Content-Type contains the expected media type, not just if it ends with it. A more robust implementation would parse the Content-Type header or use substring matching:

```diff
     if proto is None and content_type:
-        if content_type.endswith(('json', 'javascript')):
+        # Split on ';' to handle parameters like 'application/json; charset=utf-8'
+        media_type = content_type.split(';')[0].strip()
+        if media_type.endswith(('json', 'javascript')):
             pass
-        elif allow_pickle and content_type.endswith('pickle'):
+        elif allow_pickle and media_type.endswith('pickle'):
             proto = Protocol.pickle
         else:
             raise TypeError(f'Unknown content-type: {content_type}')
```