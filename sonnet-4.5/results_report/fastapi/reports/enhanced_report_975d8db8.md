# Bug Report: fastapi.encoders.jsonable_encoder Incorrectly Excludes Entire Parent Key with Nested Exclude

**Target**: `fastapi.encoders.jsonable_encoder`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When using dict-form `exclude` parameter with nested keys (e.g., `exclude={"user": {"password"}}`), `jsonable_encoder` incorrectly excludes the entire parent key instead of just the specified nested fields when encoding plain dict objects.

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

if __name__ == "__main__":
    test_nested_exclude_property()
```

<details>

<summary>
**Failing input**: `obj={'public': '', 'nested': {'': 0, '0': 0}}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 31, in <module>
    test_nested_exclude_property()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 6, in test_nested_exclude_property
    obj=st.fixed_dictionaries({
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 21, in test_nested_exclude_property
    assert "nested" in result, f"nested should be in result, got {result}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: nested should be in result, got {'public': ''}
Falsifying example: test_nested_exclude_property(
    obj={'public': '', 'nested': {'': 0, '0': 0}},
)
```
</details>

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

print("Original object:")
print(obj)
print()

print("Result with exclude={\"user\": {\"password\"}}:")
result = jsonable_encoder(obj, exclude={"user": {"password"}})
print(result)
print()

print("Expected result:")
expected = {
    'user': {'name': 'John', 'email': 'john@example.com'},
    'post': {'title': 'Hello', 'content': 'World'}
}
print(expected)
print()

print("Bug: The entire 'user' key is missing instead of just 'password' being excluded!")
```

<details>

<summary>
Output demonstrating the bug
</summary>
```
Original object:
{'user': {'name': 'John', 'password': 'secret', 'email': 'john@example.com'}, 'post': {'title': 'Hello', 'content': 'World'}}

Result with exclude={"user": {"password"}}:
{'post': {'title': 'Hello', 'content': 'World'}}

Expected result:
{'user': {'name': 'John', 'email': 'john@example.com'}, 'post': {'title': 'Hello', 'content': 'World'}}

Bug: The entire 'user' key is missing instead of just 'password' being excluded!
```
</details>

## Why This Is A Bug

The function violates expected behavior and its own documentation in several ways:

1. **Documentation mismatch**: The docstring states that `exclude` accepts "Pydantic's `exclude` parameter" which supports nested dict-form exclusion. Pydantic's exclude parameter with `{"user": {"password"}}` excludes only the password field within user, not the entire user object.

2. **Inconsistent behavior**: The same dict-form exclude works correctly for Pydantic models and dataclasses (lines 243-254) but fails for plain dicts.

3. **Logic error in implementation**: At line 270 in `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/fastapi/encoders.py`, the code does:
   ```python
   if exclude is not None:
       allowed_keys -= set(exclude)
   ```
   When `exclude` is `{"user": {"password"}}`, `set(exclude)` returns `{"user"}`, which removes the entire "user" key from `allowed_keys`. This is clearly incorrect - the code should handle nested exclusion by passing the nested exclude dict to the recursive call.

4. **Type hints allow dict-form**: The `exclude` parameter type hint is `Optional[IncEx]` where `IncEx` allows both sets and dicts, indicating dict-form exclude should be supported.

5. **No reasonable interpretation**: There is no reasonable use case where `exclude={"user": {"password"}}` would mean "exclude the entire user object". If that were the intent, the user would simply use `exclude={"user"}`.

## Relevant Context

The bug occurs specifically in the plain dict handling branch of `jsonable_encoder` (lines 264-298). The function correctly handles nested exclude for Pydantic models (via `_model_dump`) and dataclasses (via recursive calls with include/exclude), but fails to propagate nested exclude/include parameters when recursively encoding dict values.

FastAPI documentation on JSON Compatible Encoder: https://fastapi.tiangolo.com/tutorial/encoder/

The issue affects anyone using `jsonable_encoder` with plain dicts who needs to exclude specific nested fields, which is a common requirement when preparing data for API responses while hiding sensitive information.

## Proposed Fix

```diff
--- a/fastapi/encoders.py
+++ b/fastapi/encoders.py
@@ -266,9 +266,15 @@ def jsonable_encoder(
         allowed_keys = set(obj.keys())
         if include is not None:
             allowed_keys &= set(include)
         if exclude is not None:
-            allowed_keys -= set(exclude)
+            if isinstance(exclude, dict):
+                # Don't remove keys that have nested excludes
+                allowed_keys -= {k for k in exclude if not isinstance(exclude[k], (set, dict))}
+            else:
+                allowed_keys -= set(exclude)
         for key, value in obj.items():
             if (
                 (
                     not sqlalchemy_safe
                     or (not isinstance(key, str))
@@ -286,11 +292,23 @@ def jsonable_encoder(
                     exclude_none=exclude_none,
                     custom_encoder=custom_encoder,
                     sqlalchemy_safe=sqlalchemy_safe,
                 )
+                # Handle nested include/exclude
+                nested_include = None
+                nested_exclude = None
+                if isinstance(include, dict) and key in include:
+                    nested_include = include[key]
+                if isinstance(exclude, dict) and key in exclude:
+                    nested_exclude = exclude[key]
+
                 encoded_value = jsonable_encoder(
                     value,
+                    include=nested_include,
+                    exclude=nested_exclude,
                     by_alias=by_alias,
                     exclude_unset=exclude_unset,
+                    exclude_defaults=exclude_defaults,
                     exclude_none=exclude_none,
                     custom_encoder=custom_encoder,
                     sqlalchemy_safe=sqlalchemy_safe,
                 )
```