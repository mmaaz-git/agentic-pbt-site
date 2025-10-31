# Bug Report: jsonable_encoder Nested Exclude Excludes Entire Parent Key

**Target**: `fastapi.encoders.jsonable_encoder`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When using dict-form `exclude` parameter with nested keys (e.g., `exclude={"user": {"password"}}`), `jsonable_encoder` incorrectly excludes the entire parent key instead of just the specified nested fields.

## Property-Based Test

```python
from fastapi.encoders import jsonable_encoder
from hypothesis import given, strategies as st, settings


@given(
    obj=st.fixed_dictionaries({
        "public": st.text(),
        "nested": st.dictionaries(st.text(), st.integers(), min_size=2, max_size=5)
    })
)
@settings(max_examples=50)
def test_nested_exclude_property(obj):
    nested_keys = list(obj["nested"].keys())
    if len(nested_keys) < 2:
        return

    exclude_key = nested_keys[0]
    result = jsonable_encoder(obj, exclude={"nested": {exclude_key}})

    assert "public" in result
    assert "nested" in result, f"nested should be in result, got {result}"

    if "nested" in result:
        assert exclude_key not in result["nested"], \
            f"{exclude_key} should be excluded from nested"
        for key in nested_keys[1:]:
            assert key in result["nested"], \
                f"{key} should still be in nested"
```

**Failing input**: Any dict with nested structure and dict-form exclude

## Reproducing the Bug

```python
from fastapi.encoders import jsonable_encoder

obj = {
    "user": {
        "name": "John",
        "password": "secret",
        "email": "john@example.com"
    },
    "post": {
        "title": "Hello",
        "content": "World"
    }
}

result = jsonable_encoder(obj, exclude={"user": {"password"}})
print(result)
```

**Actual output**:
```python
{'post': {'title': 'Hello', 'content': 'World'}}
```

**Expected output**:
```python
{
    'user': {'name': 'John', 'email': 'john@example.com'},
    'post': {'title': 'Hello', 'content': 'World'}
}
```

## Why This Is A Bug

The function should support nested exclude/include dicts as documented in Pydantic. When `exclude={"user": {"password"}}` is provided, only the `"password"` field within `"user"` should be excluded, not the entire `"user"` key.

Currently, the implementation does:
```python
allowed_keys -= set(exclude)  # This gets {"user"} from {"user": {"password"}}
```

This excludes the entire `"user"` key instead of passing the nested exclusion down to the recursive call.

## Fix

The fix requires properly handling nested include/exclude dicts for plain dict objects:

```diff
 if isinstance(obj, dict):
     encoded_dict = {}
     allowed_keys = set(obj.keys())
     if include is not None:
         allowed_keys &= set(include)
     if exclude is not None:
         allowed_keys -= set(exclude)
     for key, value in obj.items():
         if (
             (
                 not sqlalchemy_safe
                 or (not isinstance(key, str))
                 or (not key.startswith("_sa"))
             )
             and (value is not None or not exclude_none)
             and key in allowed_keys
         ):
             encoded_key = jsonable_encoder(
                 key,
                 by_alias=by_alias,
                 exclude_unset=exclude_unset,
                 exclude_none=exclude_none,
                 custom_encoder=custom_encoder,
                 sqlalchemy_safe=sqlalchemy_safe,
             )
+            # Handle nested include/exclude
+            nested_include = None
+            nested_exclude = None
+            if isinstance(include, dict) and key in include:
+                nested_include = include[key]
+            if isinstance(exclude, dict) and key in exclude:
+                nested_exclude = exclude[key]
+
             encoded_value = jsonable_encoder(
                 value,
+                include=nested_include,
+                exclude=nested_exclude,
                 by_alias=by_alias,
                 exclude_unset=exclude_unset,
                 exclude_none=exclude_none,
                 custom_encoder=custom_encoder,
                 sqlalchemy_safe=sqlalchemy_safe,
             )
             encoded_dict[encoded_key] = encoded_value
     return encoded_dict
```