# Bug Report: xarray.util.CombineKwargDefault Hash Instability

**Target**: `xarray.util.deprecation_helpers.CombineKwargDefault`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombineKwargDefault` class violates Python's hash contract by allowing an object's hash to change during its lifetime when the global `OPTIONS["use_new_combine_kwarg_defaults"]` setting is modified.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS


@given(
    name=st.text(min_size=1),
    old=st.text(),
    new=st.one_of(st.none(), st.text())
)
@settings(max_examples=1000)
def test_combine_kwarg_hash_stable_across_options_change(name, old, new):
    obj = CombineKwargDefault(name=name, old=old, new=new)

    original_setting = OPTIONS["use_new_combine_kwarg_defaults"]
    hash1 = hash(obj)

    OPTIONS["use_new_combine_kwarg_defaults"] = not original_setting
    hash2 = hash(obj)

    OPTIONS["use_new_combine_kwarg_defaults"] = original_setting

    assert hash1 == hash2
```

**Failing input**: `name='0'`, `old=''`, `new=None`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS

obj = CombineKwargDefault(name='test', old='old_val', new='new_val')

original_setting = OPTIONS["use_new_combine_kwarg_defaults"]
hash1 = hash(obj)

OPTIONS["use_new_combine_kwarg_defaults"] = not original_setting
hash2 = hash(obj)

print(f"Hash with OPTIONS={original_setting}: {hash1}")
print(f"Hash with OPTIONS={not original_setting}: {hash2}")
print(f"Bug: Hash changed!")

OPTIONS["use_new_combine_kwarg_defaults"] = original_setting
```

## Why This Is A Bug

Python's data model requires that an object's hash value remain constant during its lifetime. From the Python documentation:

> "If a class defines mutable objects and implements an __eq__() method, it should not implement __hash__(), since the implementation of hashable collections requires that a key's hash value is immutable."

The current implementation computes `__hash__` based on `self._value`, which is a property that depends on the global `OPTIONS` setting:

```python
def __hash__(self) -> int:
    return hash(self._value)

@property
def _value(self) -> str | None:
    return self._new if OPTIONS["use_new_combine_kwarg_defaults"] else self._old
```

This can cause objects to be "lost" in dictionaries or sets if the OPTIONS setting changes after the object is inserted, because the hash used for lookup will differ from the hash used for insertion.

## Fix

The hash should be based on the immutable construction parameters rather than the mutable `_value` property:

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

This ensures the hash remains constant throughout the object's lifetime, while `__eq__` can still use `_value` for equality comparisons.