# Bug Report: xarray.compat.array_api_compat.get_array_namespace Unhashable Namespace Crash

**Target**: `xarray.compat.array_api_compat.get_array_namespace`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_array_namespace` function crashes with `TypeError: unhashable type` when arrays return namespace objects that are not hashable via their `__array_namespace__()` method. This violates the Array API standard which does not require namespace objects to be hashable.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.compat.array_api_compat import get_array_namespace


class ArrayWithNamespace:
    def __init__(self, namespace_obj):
        self._namespace = namespace_obj

    def __array_namespace__(self):
        return self._namespace


class UnhashableNamespace:
    __name__ = "custom"
    __hash__ = None


@given(st.integers(min_value=1, max_value=5))
def test_get_array_namespace_hashability(n_arrays):
    unhashable_ns = UnhashableNamespace()
    arrays = [ArrayWithNamespace(unhashable_ns) for _ in range(n_arrays)]

    get_array_namespace(*arrays)
```

**Failing input**: Any array with `__array_namespace__()` returning an unhashable object

## Reproducing the Bug

```python
from xarray.compat.array_api_compat import get_array_namespace


class ArrayWithNamespace:
    def __init__(self, namespace_obj):
        self._namespace = namespace_obj

    def __array_namespace__(self):
        return self._namespace


class UnhashableNamespace:
    __name__ = "custom_namespace"
    __hash__ = None


unhashable_ns = UnhashableNamespace()
arr1 = ArrayWithNamespace(unhashable_ns)
arr2 = ArrayWithNamespace(unhashable_ns)

get_array_namespace(arr1, arr2)
```

Output:
```
TypeError: unhashable type: 'UnhashableNamespace'
```

## Why This Is A Bug

The Array API standard (which `__array_namespace__` is part of) does not require namespace objects to be hashable. The function should work with any valid namespace object, not just hashable ones.

Line 62 in `array_api_compat.py` creates a set of namespaces:
```python
namespaces = {_get_single_namespace(t) for t in values}
```

This fails when namespace objects are not hashable, which is a valid case according to the standard.

## Fix

Replace the set-based deduplication with a list-based approach that uses object identity (`is`) for comparison:

```diff
--- a/xarray/compat/array_api_compat.py
+++ b/xarray/compat/array_api_compat.py
@@ -59,8 +59,13 @@ def get_array_namespace(*values):
         else:
             return np

-    namespaces = {_get_single_namespace(t) for t in values}
-    non_numpy = namespaces - {np}
+    namespaces = []
+    for t in values:
+        ns = _get_single_namespace(t)
+        if not any(ns is existing for existing in namespaces):
+            namespaces.append(ns)
+
+    non_numpy = [ns for ns in namespaces if ns is not np]

     if len(non_numpy) > 1:
         names = [module.__name__ for module in non_numpy]
```