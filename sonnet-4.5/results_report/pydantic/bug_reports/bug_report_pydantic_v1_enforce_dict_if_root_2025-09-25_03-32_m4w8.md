# Bug Report: pydantic.v1 Custom Root Dict Models Double-Wrapping

**Target**: `pydantic.v1.main.BaseModel._enforce_dict_if_root`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_enforce_dict_if_root` method incorrectly double-wraps dictionaries that are already in the correct format when the model has a custom root type with a mapping-like shape (Dict, Mapping, DefaultDict, Counter).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.v1 import BaseModel
from typing import Dict

class DictRootModel(BaseModel):
    __root__: Dict[str, int]

@given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers()))
def test_parse_obj_idempotent(data):
    wrapped = {"__root__": data}

    model1 = DictRootModel.parse_obj(data)
    model2 = DictRootModel.parse_obj(wrapped)

    assert model1 == model2, \
        f"parse_obj should handle both wrapped and unwrapped dicts identically"
    assert model1.__root__ == data
    assert model2.__root__ == data
```

**Failing input**: `{"__root__": {"a": 1}}` (already wrapped dict)

## Reproducing the Bug

```python
from pydantic.v1 import BaseModel
from typing import Dict

class DictRootModel(BaseModel):
    __root__: Dict[str, int]

already_wrapped = {"__root__": {"a": 1, "b": 2}}

result = DictRootModel._enforce_dict_if_root(already_wrapped)

assert result == {"__root__": {"__root__": {"a": 1, "b": 2}}}

model = DictRootModel.parse_obj(already_wrapped)

assert model.__root__ == {"__root__": {"a": 1, "b": 2}}
```

## Why This Is A Bug

The logic in `_enforce_dict_if_root` (pydantic/v1/main.py:513-521) has a flawed conditional:

```python
if cls.__custom_root_type__ and (
    not (isinstance(obj, dict) and obj.keys() == {ROOT_KEY})
    and not (isinstance(obj, BaseModel) and obj.__fields__.keys() == {ROOT_KEY})
    or cls.__fields__[ROOT_KEY].shape in MAPPING_LIKE_SHAPES
):
    return {ROOT_KEY: obj}
```

Due to operator precedence, this evaluates as:

```python
if cls.__custom_root_type__ and (
    ((not A) and (not B)) or C
):
```

When `C` (mapping-like shape) is True, the function wraps the object **regardless** of whether it's already in the correct `{ROOT_KEY: data}` format. This causes double-wrapping for already-wrapped dicts.

**Trace for `obj = {"__root__": {"a": 1}}`:**
1. `A = isinstance(obj, dict) and obj.keys() == {ROOT_KEY}` → True
2. `not A` → False
3. `B = isinstance(obj, BaseModel) and ...` → False
4. `not B` → True
5. `C = shape in MAPPING_LIKE_SHAPES` → True
6. Condition: `(False and True) or True` → True
7. Result: Wraps to `{"__root__": {"__root__": {"a": 1}}}`

## Fix

Add an early return for already-correctly-formatted inputs:

```diff
--- a/pydantic/v1/main.py
+++ b/pydantic/v1/main.py
@@ -512,11 +512,14 @@ class BaseModel(Representation, metaclass=ModelMetaclass):
     @classmethod
     def _enforce_dict_if_root(cls, obj: Any) -> Any:
+        # If already in correct format, don't re-wrap
+        if isinstance(obj, dict) and obj.keys() == {ROOT_KEY}:
+            return obj
+
         if cls.__custom_root_type__ and (
-            not (isinstance(obj, dict) and obj.keys() == {ROOT_KEY})
-            and not (isinstance(obj, BaseModel) and obj.__fields__.keys() == {ROOT_KEY})
+            not (isinstance(obj, BaseModel) and obj.__fields__.keys() == {ROOT_KEY})
             or cls.__fields__[ROOT_KEY].shape in MAPPING_LIKE_SHAPES
         ):
             return {ROOT_KEY: obj}
         else:
             return obj
```

Alternatively, fix the conditional logic:

```diff
--- a/pydantic/v1/main.py
+++ b/pydantic/v1/main.py
@@ -513,10 +513,10 @@ class BaseModel(Representation, metaclass=ModelMetaclass):
     def _enforce_dict_if_root(cls, obj: Any) -> Any:
         if cls.__custom_root_type__ and (
-            not (isinstance(obj, dict) and obj.keys() == {ROOT_KEY})
-            and not (isinstance(obj, BaseModel) and obj.__fields__.keys() == {ROOT_KEY})
-            or cls.__fields__[ROOT_KEY].shape in MAPPING_LIKE_SHAPES
+            (not (isinstance(obj, dict) and obj.keys() == {ROOT_KEY})
+             and not (isinstance(obj, BaseModel) and obj.__fields__.keys() == {ROOT_KEY}))
+            and cls.__fields__[ROOT_KEY].shape in MAPPING_LIKE_SHAPES
         ):
             return {ROOT_KEY: obj}
         else:
             return obj
```

The second fix changes `or` to `and`, so mapping-like shapes are only wrapped if they're not already in the correct format.