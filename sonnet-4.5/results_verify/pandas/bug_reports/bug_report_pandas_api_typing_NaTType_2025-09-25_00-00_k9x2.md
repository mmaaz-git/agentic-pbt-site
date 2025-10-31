# Bug Report: pandas.api.typing.NaTType Singleton Violation

**Target**: `pandas.api.typing.NaTType`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

NaTType() creates a new instance on each call instead of returning the singleton `pd.NaT`, violating the singleton pattern that NAType correctly implements.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.api.typing import NaTType


@settings(max_examples=100)
@given(st.integers(min_value=1, max_value=100))
def test_nattype_singleton_multiple_calls(n):
    instances = [NaTType() for _ in range(n)]
    first = instances[0]
    assert all(inst is first for inst in instances), "NaTType() should always return the same singleton instance"
```

**Failing input**: `n=2`

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.typing import NaTType, NAType

nat1 = NaTType()
nat2 = NaTType()

print(f"NaTType() is NaTType(): {nat1 is nat2}")

print(f"\nComparison with NAType (correct behavior):")
na1 = NAType()
na2 = NAType()
print(f"NAType() is NAType(): {na1 is na2}")

print(f"\nComparison with canonical pd.NaT:")
print(f"pd.NaT is pd.NaT: {pd.NaT is pd.NaT}")
print(f"NaTType() is pd.NaT: {nat1 is pd.NaT}")
```

Output:
```
NaTType() is NaTType(): False

Comparison with NAType (correct behavior):
NAType() is NAType(): True

Comparison with canonical pd.NaT:
pd.NaT is pd.NaT: True
NaTType() is pd.NaT: False
```

## Why This Is A Bug

1. **Singleton pattern violation**: NaTType represents a singleton missing value indicator (like NaN for time data), similar to NAType. NAType correctly returns the same singleton instance (`pd.NA`) on every call, but NaTType creates new instances.

2. **Identity comparison failures**: Code using `x is pd.NaT` will fail if `x = NaTType()`, even though both represent the same conceptual "not-a-time" value.

3. **API inconsistency**: Within the same module, NAType and NaTType have identical roles (missing value indicators) but behave differently - one is a proper singleton, the other is not.

## Fix

NaTType's `__new__` method should return the canonical `pd.NaT` singleton, similar to how NAType returns `pd.NA`. The implementation should look like:

```diff
--- a/pandas/_libs/tslibs/nattype.pyx
+++ b/pandas/_libs/tslibs/nattype.pyx
@@ -XXX,X +XXX,X @@ cdef class NaTType:
     def __new__(cls):
-        # Currently creates new instances
-        cdef NaTType obj = <NaTType>object.__new__(cls)
-        return obj
+        # Return the singleton instance
+        return c_NaT
```

Note: The exact fix depends on the Cython implementation details, but the key is to return the existing singleton `c_NaT` or equivalent instead of creating a new instance.