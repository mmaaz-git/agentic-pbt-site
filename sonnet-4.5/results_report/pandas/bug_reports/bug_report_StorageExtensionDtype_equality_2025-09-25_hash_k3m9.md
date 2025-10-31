# Bug Report: StorageExtensionDtype Equality and Hash Violations

**Target**: `pandas.core.dtypes.base.StorageExtensionDtype`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`StorageExtensionDtype.__eq__` violates two fundamental properties: (1) the Python hash-equality contract that requires equal objects to have equal hashes, and (2) equality transitivity. This occurs because `__eq__` considers dtypes with different storage parameters equal to the same string, while `__hash__` differentiates them.

## Property-Based Test

```python
from pandas.core.dtypes.base import StorageExtensionDtype
from hypothesis import given, strategies as st, settings


class TestStorageDtype(StorageExtensionDtype):
    name = "test_storage"

    @classmethod
    def construct_array_type(cls):
        return None


@given(storage=st.one_of(st.none(), st.text(min_size=1, max_size=20)))
@settings(max_examples=500)
def test_storage_dtype_hash_consistency_with_string(storage):
    dtype = TestStorageDtype(storage=storage)

    if dtype == dtype.name:
        assert hash(dtype) == hash(dtype.name)


@given(
    storage1=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
    storage2=st.one_of(st.none(), st.text(min_size=1, max_size=20))
)
@settings(max_examples=500)
def test_storage_dtype_equality_transitivity(storage1, storage2):
    dtype1 = TestStorageDtype(storage=storage1)
    dtype2 = TestStorageDtype(storage=storage2)
    name_str = dtype1.name

    if dtype1 == name_str and name_str == dtype2:
        assert dtype1 == dtype2
```

**Failing inputs**:
- Hash consistency: `storage=None` (or any value)
- Transitivity: `storage1=None, storage2='0'` (or any two different values)

## Reproducing the Bug

```python
from pandas.core.dtypes.base import StorageExtensionDtype


class MyStorageDtype(StorageExtensionDtype):
    name = "mystorage"

    @classmethod
    def construct_array_type(cls):
        return None


dtype1 = MyStorageDtype(storage="pyarrow")
dtype2 = MyStorageDtype(storage="python")
name = "mystorage"

print(f"dtype1 == name: {dtype1 == name}")
print(f"hash(dtype1) == hash(name): {hash(dtype1) == hash(name)}")

print(f"\ndtype1 == name: {dtype1 == name}")
print(f"name == dtype2: {name == dtype2}")
print(f"dtype1 == dtype2: {dtype1 == dtype2}")
```

## Why This Is A Bug

This violates two critical invariants:

1. **Hash-Equality Contract**: Python requires that if `a == b`, then `hash(a) == hash(b)`. This is essential for correct behavior in sets and dictionaries. When `dtype == name_string` is True but their hashes differ, dtypes may not be found in sets/dicts where they should be.

2. **Transitivity**: Equality must be transitive: if `a == b` and `b == c`, then `a == c`. This is a fundamental mathematical property. When `dtype1 == "mystorage"` and `"mystorage" == dtype2` are both True, but `dtype1 != dtype2`, it violates user expectations and can lead to confusing behavior.

The root cause is at lines 464-467 in `pandas/core/dtypes/base.py`:

```python
def __eq__(self, other: object) -> bool:
    if isinstance(other, str) and other == self.name:
        return True  # Returns True regardless of storage parameter
    return super().__eq__(other)
```

This makes all instances with the same name equal to the name string, but `__hash__` (line 469-471) uses `super().__hash__()` which depends on the storage parameter via `_metadata`.

## Fix

The fix depends on the intended behavior:

**Option 1**: Remove string equality entirely (most conservative):

```diff
--- a/pandas/core/dtypes/base.py
+++ b/pandas/core/dtypes/base.py
@@ -463,9 +463,6 @@ class StorageExtensionDtype(ExtensionDtype):
         return self.name

     def __eq__(self, other: object) -> bool:
-        if isinstance(other, str) and other == self.name:
-            return True
         return super().__eq__(other)

     def __hash__(self) -> int:
```

**Option 2**: Make hash consistent with string equality by hashing only the name when storage is None:

```diff
--- a/pandas/core/dtypes/base.py
+++ b/pandas/core/dtypes/base.py
@@ -468,8 +468,10 @@ class StorageExtensionDtype(ExtensionDtype):
         return super().__eq__(other)

     def __hash__(self) -> int:
-        # custom __eq__ so have to override __hash__
-        return super().__hash__()
+        if self.storage is None:
+            return hash(self.name)
+        else:
+            return super().__hash__()
```

**Recommendation**: Option 1 is safer and simpler. If string equality is needed for convenience, it should only return True when the string can construct an equivalent dtype (i.e., one with matching metadata), which is already handled by the parent class's `__eq__` implementation.