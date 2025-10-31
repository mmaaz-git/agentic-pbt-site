# Bug Report: CombineKwargDefault Hash Instability

**Target**: `xarray.util.deprecation_helpers.CombineKwargDefault`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombineKwargDefault` class violates Python's hash stability requirement: an object's hash changes when the global `OPTIONS["use_new_combine_kwarg_defaults"]` setting changes, breaking set/dict membership and violating the fundamental invariant that an object's hash must remain constant during its lifetime.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS


@given(st.sampled_from(["all", "minimal", "exact"]))
@settings(max_examples=100)
def test_hash_stability_across_options_change(val):
    obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

    original_option = OPTIONS["use_new_combine_kwarg_defaults"]

    try:
        OPTIONS["use_new_combine_kwarg_defaults"] = False
        hash1 = hash(obj)

        OPTIONS["use_new_combine_kwarg_defaults"] = True
        hash2 = hash(obj)

        assert hash1 == hash2
    finally:
        OPTIONS["use_new_combine_kwarg_defaults"] = original_option
```

**Failing input**: Any CombineKwargDefault instance with different old/new values

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS

obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

OPTIONS["use_new_combine_kwarg_defaults"] = False
s = {obj}
assert obj in s

OPTIONS["use_new_combine_kwarg_defaults"] = True
assert obj in s
```

The second assertion fails because the object's hash changed, so Python can't find it in the set.

## Why This Is A Bug

Python's data model requires that an object's hash must remain constant during its lifetime (see [Python docs](https://docs.python.org/3/reference/datamodel.html#object.__hash__)). The current implementation violates this by computing hash from `self._value`, which changes based on global state. This causes objects to be lost in sets and dictionaries when the global option changes, leading to incorrect program behavior.

## Fix

The hash should be based on immutable properties of the object (name, old, new) rather than the computed `_value` which depends on mutable global state:

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