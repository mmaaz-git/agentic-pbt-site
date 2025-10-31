# Bug Report: pandas.api.typing.NaTType Constructor Does Not Return Singleton

**Target**: `pandas.api.typing.NaTType`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

NaTType() constructor creates new instances on each call instead of returning the singleton pd.NaT, inconsistent with NAType's singleton behavior.

## Property-Based Test

```python
import pandas as pd
from pandas.api.typing import NaTType
from hypothesis import given, strategies as st


@given(st.integers(min_value=0, max_value=100))
def test_nattype_singleton_property(n):
    instances = [NaTType() for _ in range(n)]
    if len(instances) > 0:
        for instance in instances:
            assert instance is pd.NaT, f"NaTType() should return the singleton pd.NaT"


def test_nattype_constructors_return_same_instance():
    nat1 = NaTType()
    nat2 = NaTType()
    assert nat1 is nat2, "Multiple NaTType() calls should return the same singleton instance"
```

**Failing input**: `n=1`

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.typing import NaTType, NAType

nat1 = NaTType()
nat2 = NaTType()
print(f"NaTType() is NaTType(): {nat1 is nat2}")
print(f"NaTType() is pd.NaT: {nat1 is pd.NaT}")

na1 = NAType()
na2 = NAType()
print(f"NAType() is NAType(): {na1 is na2}")
print(f"NAType() is pd.NA: {na1 is pd.NA}")
```

Output:
```
NaTType() is NaTType(): False
NaTType() is pd.NaT: False
NAType() is NAType(): True
NAType() is pd.NA: True
```

## Why This Is A Bug

1. **API Inconsistency**: NAType correctly implements singleton behavior via `__new__`, returning the same `pd.NA` instance on every call. NaTType should behave the same way but doesn't.

2. **Unexpected Behavior**: Both NaTType and NAType are exported in `pandas.api.typing` as type-hinting helpers. Users would reasonably expect consistent constructor behavior between these similar types.

3. **Multiple Instances**: While `pd.NaT` itself is a singleton, calling `NaTType()` creates distinct instances that are not identical to `pd.NaT` or each other, violating the singleton pattern.

## Fix

NaTType should implement `__new__` to return the singleton instance, similar to NAType. The implementation would look like:

```diff
--- a/pandas/_libs/tslibs/nattype.pyx
+++ b/pandas/_libs/tslibs/nattype.pyx
@@ -XX,X +XX,X @@ class NaTType:
+    def __new__(cls):
+        return NaT
```

This ensures `NaTType()` returns the `pd.NaT` singleton, making it consistent with NAType's behavior.
