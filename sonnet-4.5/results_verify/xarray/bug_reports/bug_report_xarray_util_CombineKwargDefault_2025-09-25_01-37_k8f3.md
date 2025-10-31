# Bug Report: xarray.util CombineKwargDefault Hash Mutability

**Target**: `xarray.util.deprecation_helpers.CombineKwargDefault`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The hash of `CombineKwargDefault` objects changes when global OPTIONS change, violating Python's requirement that object hashes must remain constant during their lifetime. This makes these objects unsuitable for use as dictionary keys or set members.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

@given(
    name=st.text(min_size=1),
    old=st.text(),
    new=st.text()
)
def test_hash_should_not_change_with_options(name, old, new):
    assume(old != new)

    obj = CombineKwargDefault(name=name, old=old, new=new)

    with set_options(use_new_combine_kwarg_defaults=False):
        hash1 = hash(obj)

    with set_options(use_new_combine_kwarg_defaults=True):
        hash2 = hash(obj)

    assert hash1 == hash2
```

**Failing input**: `name='0', old='', new='0'`

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

with set_options(use_new_combine_kwarg_defaults=False):
    hash1 = hash(obj)
    s = {obj}
    assert obj in s

with set_options(use_new_combine_kwarg_defaults=True):
    hash2 = hash(obj)
    print(f"Hash changed: {hash1} -> {hash2}")
    print(f"Object in set: {obj in s}")
```

Output:
```
Hash changed: 5988906273155277687 -> 0
Object in set: False
```

## Why This Is A Bug

The `__hash__` method (line 180-181 in deprecation_helpers.py) returns `hash(self._value)`, where `_value` is a property that depends on global OPTIONS:

```python
@property
def _value(self) -> str | None:
    return self._new if OPTIONS["use_new_combine_kwarg_defaults"] else self._old

def __hash__(self) -> int:
    return hash(self._value)
```

This violates Python's requirement that "the hash value of an object must not change during its lifetime" (Python documentation). When the hash changes:
1. Objects cannot be found in sets after being added
2. Objects cannot be retrieved from dictionaries after being stored
3. This causes silent data corruption

## Fix

The hash should be based on immutable instance data, not the mutable `_value` property:

```diff
diff --git a/xarray/util/deprecation_helpers.py b/xarray/util/deprecation_helpers.py
index 1234567..abcdefg 100644
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -177,8 +177,8 @@ class CombineKwargDefault:
     def _value(self) -> str | None:
         return self._new if OPTIONS["use_new_combine_kwarg_defaults"] else self._old

-    def __hash__(self) -> int:
-        return hash(self._value)
+    def __hash__(self) -> int:
+        return hash((self._name, self._old, self._new))

     def __dask_tokenize__(self) -> object:
         from dask.base import normalize_token
```