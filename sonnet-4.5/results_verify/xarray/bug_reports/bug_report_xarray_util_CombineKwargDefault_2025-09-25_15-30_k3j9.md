# Bug Report: xarray.util.CombineKwargDefault Hash Instability

**Target**: `xarray.util.deprecation_helpers.CombineKwargDefault`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombineKwargDefault` class has an unstable hash that changes when the global `OPTIONS["use_new_combine_kwarg_defaults"]` setting is modified. This violates Python's hash contract and causes objects to become unfindable in sets and dictionaries after the OPTIONS change.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS

@given(
    name=st.text(min_size=1, max_size=20),
    old=st.text(min_size=1, max_size=20),
    new=st.text(min_size=1, max_size=20),
)
def test_combine_kwarg_in_set_with_options_change(name, old, new):
    assume(old != new)
    obj = CombineKwargDefault(name=name, old=old, new=new)
    s = {obj}
    assert obj in s

    original_setting = OPTIONS["use_new_combine_kwarg_defaults"]
    try:
        OPTIONS["use_new_combine_kwarg_defaults"] = not original_setting
        assert obj in s
    finally:
        OPTIONS["use_new_combine_kwarg_defaults"] = original_setting
```

**Failing input**: `name="test", old="old_value", new="new_value"`

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS

obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

s = {obj}
print(f"Object in set: {obj in s}")

original_setting = OPTIONS["use_new_combine_kwarg_defaults"]
OPTIONS["use_new_combine_kwarg_defaults"] = not original_setting

print(f"Object still in set after OPTIONS change: {obj in s}")
print(f"Expected: True, Actual: False")

OPTIONS["use_new_combine_kwarg_defaults"] = original_setting
```

## Why This Is A Bug

Python's hash contract requires that:
1. The hash of an object must never change during its lifetime
2. If `a == b`, then `hash(a) == hash(b)`

The current implementation violates requirement #1 by computing the hash from `_value`, which changes based on the global `OPTIONS` setting:

```python
def __hash__(self) -> int:
    return hash(self._value)

@property
def _value(self) -> str | None:
    return self._new if OPTIONS["use_new_combine_kwarg_defaults"] else self._old
```

When `OPTIONS["use_new_combine_kwarg_defaults"]` changes, `_value` changes, and therefore `hash(obj)` changes. This breaks sets and dictionaries that contain `CombineKwargDefault` objects.

## Fix

The hash should be based on immutable attributes that don't change during the object's lifetime:

```diff
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -178,7 +178,7 @@ class CombineKwargDefault:
         return self._new if OPTIONS["use_new_combine_kwarg_defaults"] else self._old

     def __hash__(self) -> int:
-        return hash(self._value)
+        return hash((self._name, self._old, self._new))

     def __dask_tokenize__(self) -> object:
         from dask.base import normalize_token
```

This ensures the hash remains stable regardless of OPTIONS changes, while still allowing `_value` to be mutable for its intended purpose.