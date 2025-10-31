# Bug Report: pandas.api.typing.NaTType Constructor Inconsistency

**Target**: `pandas.api.typing.NaTType`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`NaTType()` constructor creates new instances instead of returning the singleton `pd.NaT`, inconsistent with `NAType()` which correctly returns the `pd.NA` singleton.

## Property-Based Test

```python
import pandas as pd
import pandas.api.typing
from hypothesis import given, strategies as st


@given(st.integers(min_value=1, max_value=10))
def test_nattype_natype_consistency(n):
    nat_instances = [pandas.api.typing.NaTType() for _ in range(n)]
    na_instances = [pandas.api.typing.NAType() for _ in range(n)]

    for na_inst in na_instances:
        assert na_inst is pd.NA, "NAType() returns pd.NA singleton"

    for nat_inst in nat_instances:
        assert nat_inst is pd.NaT, "NaTType() should return pd.NaT singleton for consistency"
```

**Failing input**: `n=1`

## Reproducing the Bug

```python
import pandas as pd
import pandas.api.typing

nat1 = pandas.api.typing.NaTType()
nat2 = pandas.api.typing.NaTType()

print(f"NaTType() is NaTType(): {nat1 is nat2}")
print(f"NaTType() is pd.NaT: {nat1 is pd.NaT}")

na1 = pandas.api.typing.NAType()
na2 = pandas.api.typing.NAType()

print(f"\nNAType() is NAType(): {na1 is na2}")
print(f"NAType() is pd.NA: {na1 is pd.NA}")

assert na1 is na2, "NAType() returns singleton"
assert na1 is pd.NA, "NAType() returns pd.NA"
assert nat1 is nat2, "NaTType() should return singleton (FAILS)"
```

Output:
```
NaTType() is NaTType(): False
NaTType() is pd.NaT: False

NAType() is NAType(): True
NAType() is pd.NA: True

AssertionError: NaTType() should return singleton (FAILS)
```

## Why This Is A Bug

Both `NAType` and `NaTType` are exposed in `pandas.api.typing` for type-hinting purposes. They should behave consistently:

1. `NAType()` correctly returns the singleton `pd.NA`
2. `NaTType()` incorrectly creates new instances instead of returning `pd.NaT`
3. This violates the principle of least surprise and API consistency

While the documentation describes NaT as "the time equivalent of NaN", the singleton behavior should be consistent with NAType. The NaN-like comparison behavior (`pd.NaT == pd.NaT` returning `False`) is separate from constructor behavior.

## Fix

The `NaTType.__new__` method should be implemented to return the singleton instance, similar to how `NAType` is implemented. Since NaTType is implemented in C (`.so` file), the fix would need to be applied in the pandas C extension code at `pandas/_libs/tslibs/nattype.pyx` or similar.

A conceptual Python-level fix would look like:

```diff
 class NaTType:
-    def __new__(cls):
-        return object.__new__(cls)
+    _instance = None
+
+    def __new__(cls):
+        if cls._instance is None:
+            cls._instance = object.__new__(cls)
+        return cls._instance
```

However, since this is a C extension, the actual fix would need to modify the Cython source to implement singleton behavior in `__new__` or similar mechanism used by the pandas C API.