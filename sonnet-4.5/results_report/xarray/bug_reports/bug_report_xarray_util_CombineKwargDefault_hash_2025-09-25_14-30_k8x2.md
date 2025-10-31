# Bug Report: xarray.util.deprecation_helpers.CombineKwargDefault Hash Mutability

**Target**: `xarray.util.deprecation_helpers.CombineKwargDefault`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombineKwargDefault.__hash__` method returns different hash values for the same object depending on global OPTIONS state, violating Python's hash invariant that an object's hash must remain constant during its lifetime. This breaks hash-based containers like sets and dictionaries.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options


@given(
    name=st.text(min_size=1),
    old=st.text(),
    new=st.one_of(st.none(), st.text())
)
@settings(max_examples=1000)
def test_combine_kwarg_default_hash_immutable(name, old, new):
    obj = CombineKwargDefault(name=name, old=old, new=new)

    hash1 = hash(obj)

    with set_options(use_new_combine_kwarg_defaults=True):
        hash2 = hash(obj)

    with set_options(use_new_combine_kwarg_defaults=False):
        hash3 = hash(obj)

    assert hash1 == hash2 == hash3
```

**Failing input**: Any input where `old != new`, e.g., `name="test"`, `old="old_val"`, `new="new_val"`

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

obj = CombineKwargDefault(name="test", old="old_val", new="new_val")

hash1 = hash(obj)

with set_options(use_new_combine_kwarg_defaults=True):
    hash2 = hash(obj)

print(f"Hash before: {hash1}")
print(f"Hash after changing OPTIONS: {hash2}")
print(f"Hashes equal: {hash1 == hash2}")

s = {obj}
with set_options(use_new_combine_kwarg_defaults=True):
    print(f"Object in set after OPTIONS change: {obj in s}")
```

Output will show that:
1. The hash changes when OPTIONS changes
2. The object can't be found in the set after OPTIONS changes

## Why This Is A Bug

This violates Python's fundamental invariant for `__hash__`:

> The only required property is that objects which compare equal have the same hash value [...] **an object's hash value should never change during its lifetime**.

From the Python documentation, if an object is added to a set or used as a dict key, changing its hash value later breaks these containers. The object may become "lost" in the container or cause other undefined behavior.

The root cause is that both `__hash__` and `__eq__` depend on `self._value`, which is a property that reads from the global `OPTIONS` dictionary:

```python
@property
def _value(self) -> str | None:
    return self._new if OPTIONS["use_new_combine_kwarg_defaults"] else self._old

def __hash__(self) -> int:
    return hash(self._value)
```

When `OPTIONS["use_new_combine_kwarg_defaults"]` changes, `self._value` changes, causing the hash to change.

## Fix

The hash should be based on the immutable attributes (`_name`, `_old`, `_new`) rather than the mutable `_value` property:

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

This ensures that:
1. The hash remains constant during the object's lifetime
2. Two `CombineKwargDefault` objects with the same name/old/new values always have the same hash
3. The hash/equality contract is maintained: objects that compare equal (under any OPTIONS setting) will have the same hash