# Bug Report: pandas.api.typing.NaTType Singleton Inconsistency

**Target**: `pandas.api.typing.NaTType`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `NaTType()` constructor creates new instances instead of returning the `pd.NaT` singleton, causing inconsistent behavior with `NAType()` and breaking pandas missing value detection.

## Property-Based Test

```python
import pandas as pd
from pandas.api.typing import NAType, NaTType
from hypothesis import given, strategies as st

@given(st.integers())
def test_nattype_singleton_inconsistency(x):
    nat = NaTType()
    assert nat is pd.NaT, "NaTType() should return pd.NaT singleton"

@given(st.integers())
def test_nattype_equality_broken(x):
    nat1 = NaTType()
    nat2 = NaTType()
    assert nat1 == nat2, "NaTType() instances should be equal"

@given(st.integers())
def test_nattype_not_recognized_as_missing(x):
    nat = NaTType()
    assert pd.isna(nat), "NaTType() should be recognized as missing by pd.isna()"
```

**Failing input**: `x=0` (any value fails)

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.typing import NAType, NaTType

nat1 = NaTType()
nat2 = NaTType()

print(f"NAType() is pd.NA: {NAType() is pd.NA}")
print(f"NaTType() is pd.NaT: {nat1 is pd.NaT}")

print(f"nat1 is nat2: {nat1 is nat2}")
print(f"nat1 == nat2: {nat1 == nat2}")
print(f"nat1 == pd.NaT: {nat1 == pd.NaT}")

print(f"pd.isna(nat1): {pd.isna(nat1)}")
print(f"pd.isna(pd.NaT): {pd.isna(pd.NaT)}")
```

Output:
```
NAType() is pd.NA: True
NaTType() is pd.NaT: False
nat1 is nat2: False
nat1 == nat2: False
nat1 == pd.NaT: False
pd.isna(nat1): False
pd.isna(pd.NaT): True
```

## Why This Is A Bug

1. **Inconsistency with NAType**: `NAType()` correctly returns the `pd.NA` singleton, but `NaTType()` creates new instances instead of returning `pd.NaT`

2. **Broken equality**: Multiple `NaTType()` calls create instances that are not equal to each other or to `pd.NaT`, violating the expected singleton behavior for missing value sentinels

3. **Missing value detection fails**: `pd.isna(NaTType())` returns `False`, meaning pandas doesn't recognize these instances as missing values, which could lead to silent data corruption

4. **API inconsistency**: Both types are exported in `pandas.api.typing` as public API, suggesting they should behave similarly

## Fix

The `NaTType.__new__` method should return the singleton `pd.NaT` instance, similar to how `NAType.__new__` returns `pd.NA`. Without access to the Cython source, the high-level fix would be:

```python
class NaTType:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

The implementation should ensure that `NaTType()` always returns the same singleton instance that is also accessible as `pd.NaT`.