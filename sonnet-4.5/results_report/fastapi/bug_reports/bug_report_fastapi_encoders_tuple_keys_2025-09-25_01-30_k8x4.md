# Bug Report: FastAPI jsonable_encoder Tuple Dict Keys

**Target**: `fastapi.encoders.jsonable_encoder`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `jsonable_encoder` function crashes with a `TypeError: unhashable type: 'list'` when encoding dictionaries with tuple keys, because it recursively encodes the keys which converts tuples to lists, and lists cannot be used as dictionary keys.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.encoders import jsonable_encoder

@given(st.dictionaries(
    st.one_of(
        st.text(min_size=1, max_size=20),
        st.integers(),
        st.lists(st.integers(), min_size=1, max_size=3).map(tuple)
    ),
    st.integers(),
    max_size=5
))
def test_dict_with_various_key_types(obj):
    result = jsonable_encoder(obj)
    assert isinstance(result, dict)
    for key in result.keys():
        hash(key)
```

**Failing input**: `{(0,): 0}`

## Reproducing the Bug

```python
from fastapi.encoders import jsonable_encoder

obj = {(0,): 0}

result = jsonable_encoder(obj)
```

This raises:
```
TypeError: unhashable type: 'list'
```

The bug occurs at line 297 in `/fastapi/encoders.py` when trying to assign `encoded_dict[encoded_key] = encoded_value`.

## Why This Is A Bug

This violates the function's contract of converting "any object to something that can be encoded in JSON." Dictionaries with tuple keys are valid Python objects that should be handled gracefully.

The root cause is in lines 281-288 of `encoders.py`:
1. Dict keys are recursively encoded via `jsonable_encoder(key, ...)`
2. Tuples are converted to lists (line 299-315)
3. Lists are unhashable and cannot be dict keys
4. Therefore `encoded_dict[encoded_key] = encoded_value` fails

Since JSON only supports string keys anyway, the encoder should convert all non-string keys to strings rather than recursively encoding them.

## Fix

```diff
--- a/fastapi/encoders.py
+++ b/fastapi/encoders.py
@@ -278,13 +278,7 @@ def jsonable_encoder(
                 and (value is not None or not exclude_none)
                 and key in allowed_keys
             ):
-                encoded_key = jsonable_encoder(
-                    key,
-                    by_alias=by_alias,
-                    exclude_unset=exclude_unset,
-                    exclude_none=exclude_none,
-                    custom_encoder=custom_encoder,
-                    sqlalchemy_safe=sqlalchemy_safe,
-                )
+                # JSON only supports string keys, so convert non-string keys to strings
+                encoded_key = str(key) if not isinstance(key, str) else key
                 encoded_value = jsonable_encoder(
                     value,
```