# Bug Report: pandas.api.typing.NaTType Singleton Violation

**Target**: `pandas.api.typing.NaTType`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

NaTType() constructor creates new instances instead of returning the singleton pd.NaT, inconsistent with NAType behavior and causing identity check failures.

## Property-Based Test

```python
import pandas.api.typing
from hypothesis import given, strategies as st


@given(st.integers(min_value=1, max_value=100))
def test_nattype_singleton_property(n):
    instances = [pandas.api.typing.NaTType() for _ in range(n)]

    for i in range(len(instances)):
        for j in range(len(instances)):
            assert instances[i] is instances[j],                 f"NaTType() should return same singleton instance (call {i} vs {j})"
```

**Failing input**: `n=2`

## Reproducing the Bug

```python
import pandas as pd
import pandas.api.typing

nat1 = pandas.api.typing.NaTType()
nat2 = pandas.api.typing.NaTType()

assert nat1 is not nat2
assert nat1 is not pd.NaT
assert nat2 is not pd.NaT

na1 = pandas.api.typing.NAType()
na2 = pandas.api.typing.NAType()

assert na1 is na2
assert na1 is pd.NA
```

## Why This Is A Bug

1. **Inconsistent behavior**: NAType() correctly returns the pd.NA singleton, but NaTType() creates new instances instead of returning pd.NaT
2. **Violates singleton pattern**: pd.NaT is a singleton value (like None), and calling its type constructor should return that singleton
3. **Memory waste**: Each NaTType() call allocates a new object unnecessarily
4. **Identity checks fail**: Code using `x is pd.NaT` will fail for instances created via NaTType()
5. **API inconsistency**: Within the same typing module, NAType and NaTType behave differently

## Fix

NaTType should implement `__new__` method similar to NAType to enforce singleton pattern:

```diff
--- a/pandas/_libs/tslibs/nattype.pyx
+++ b/pandas/_libs/tslibs/nattype.pyx
@@ -24,6 +24,12 @@ cdef class NaTType:

     _value = NPY_NAT

+    def __new__(cls):
+        if NaT is None:
+            return object.__new__(cls)
+        else:
+            return NaT
+
     def __hash__(NaTType self):
         return NPY_NAT
```
