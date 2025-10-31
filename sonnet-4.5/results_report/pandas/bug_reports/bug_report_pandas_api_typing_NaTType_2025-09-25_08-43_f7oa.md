# Bug Report: pandas.api.typing.NaTType Creates Unrecognized NaT Instances

**Target**: `pandas.api.typing.NaTType`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Calling `NaTType()` creates new NaT-like instances that are not recognized by `pd.isna()` and are not identical to the `pd.NaT` singleton, unlike the correctly-behaving `NAType()` which returns the `pd.NA` singleton.

## Property-Based Test

```python
import pandas as pd
import pandas.api.typing as pat
from hypothesis import given, strategies as st


@given(st.integers(min_value=1, max_value=100))
def test_nattype_returns_singleton(n):
    """
    Property: NaTType() should return the same singleton instance as pd.NaT.

    This tests that calling NaTType() multiple times returns the pd.NaT singleton,
    not new instances. This is important because:
    1. pd.NaT is designed as a singleton
    2. pd.isna() and other pandas functions expect the singleton
    3. Identity checks (is) should work
    """
    instances = [pat.NaTType() for _ in range(n)]

    for instance in instances:
        assert instance is pd.NaT, f"NaTType() should return pd.NaT singleton, got different object"
        assert pd.isna(instance), f"pd.isna() should recognize NaTType() instances"
```

**Failing input**: `n=1` (or any value)

## Reproducing the Bug

```python
import pandas as pd
import pandas.api.typing as pat

nat_from_call = pat.NaTType()

print(f"nat_from_call is pd.NaT: {nat_from_call is pd.NaT}")
print(f"pd.isna(nat_from_call): {pd.isna(nat_from_call)}")

s = pd.Series([nat_from_call, pd.NaT])
print(f"\nSeries.isna():\n{s.isna()}")
```

**Output:**
```
nat_from_call is pd.NaT: False
pd.isna(nat_from_call): False

Series.isna():
0    False
1     True
dtype: bool
```

Only the real `pd.NaT` singleton is recognized as missing, not the instance created by `NaTType()`.

## Why This Is A Bug

1. **Inconsistent with NAType**: `NAType()` correctly returns the `pd.NA` singleton, but `NaTType()` creates new instances instead of returning `pd.NaT`

2. **Violates singleton pattern**: `pd.NaT` is designed as a singleton for missing datetime values, but `NaTType()` creates objects that look like NaT but aren't recognized by pandas

3. **Breaks pandas operations**: Instances created by `NaTType()` are not recognized by `pd.isna()`, causing incorrect missing value detection

4. **API confusion**: Since `NaTType` is exported in `pandas.api.typing` for type-hinting purposes, users might call it expecting to get a valid NaT value

## Fix

The `NaTType.__new__()` method should return the singleton `pd.NaT` instead of creating new instances. This would match the behavior of `NAType`:

```python
class NaTType:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

Alternatively, if `NaTType` is only meant for type-hinting and should never be instantiated, add a check to prevent instantiation:

```python
def __new__(cls):
    raise TypeError("NaTType should not be instantiated. Use pd.NaT instead.")
```