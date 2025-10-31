# Bug Report: xarray.util.CombineKwargDefault Hash Invariant Violation

**Target**: `xarray.util.deprecation_helpers.CombineKwargDefault`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombineKwargDefault.__hash__()` method violates Python's hash invariant by returning different values for the same object when global `OPTIONS["use_new_combine_kwarg_defaults"]` changes. This causes objects to become inaccessible when used as dictionary keys or in sets.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS


@given(
    name=st.text(min_size=1),
    old=st.text(),
    new=st.text(),
)
@settings(max_examples=100)
def test_hash_remains_constant_during_object_lifetime(name, old, new):
    """
    Property: An object's hash must remain constant during its lifetime.
    """
    obj = CombineKwargDefault(name=name, old=old, new=new)

    original_hash = hash(obj)
    original_option = OPTIONS["use_new_combine_kwarg_defaults"]

    OPTIONS["use_new_combine_kwarg_defaults"] = not original_option
    new_hash = hash(obj)
    OPTIONS["use_new_combine_kwarg_defaults"] = original_option

    assert original_hash == new_hash, (
        f"Hash changed when global OPTIONS changed! "
        f"Before: {original_hash}, After: {new_hash}. "
        f"This violates Python's hash invariant."
    )
```

**Failing input**: Any `CombineKwargDefault` object where `old != new`

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS

obj = CombineKwargDefault(name="test", old="old_val", new="new_val")

d = {obj: "stored_value"}
print(d[obj])

OPTIONS["use_new_combine_kwarg_defaults"] = True

try:
    print(d[obj])
except KeyError:
    print("Object lost! Hash changed, so dict lookup fails.")
```

## Why This Is A Bug

Python's data model requires that an object's hash value must remain constant during its lifetime (as stated in the [Python documentation](https://docs.python.org/3/reference/datamodel.html#object.__hash__)).

The current implementation violates this because:

1. `__hash__()` returns `hash(self._value)`
2. `_value` is a property that returns `self._new` or `self._old` based on the **global** `OPTIONS["use_new_combine_kwarg_defaults"]`
3. When this global option changes, the hash changes for existing objects

This causes severe issues:
- Objects stored as dict keys become inaccessible
- Objects in sets can't be found
- Unpredictable behavior in any code using these objects

## Fix

The hash should be based on the object's immutable attributes (`_name`, `_old`, `_new`), not on the dynamic `_value` property:

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

This ensures the hash remains constant regardless of global configuration changes.