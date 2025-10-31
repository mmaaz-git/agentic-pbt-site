# Bug Report: pandas.api.typing.NaTType Constructor Inconsistency

**Target**: `pandas.api.typing.NaTType`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`NaTType()` creates new distinct instances instead of returning the singleton `pd.NaT`, unlike the consistent behavior of `NAType()` which returns the singleton `pd.NA`.

## Property-Based Test

```python
import pandas as pd
import pandas.api.typing as pat
from hypothesis import given, strategies as st, settings

@given(st.integers(min_value=1, max_value=100))
@settings(max_examples=100)
def test_nattype_singleton_property(n):
    instances = [pat.NaTType() for _ in range(n)]

    for i, instance in enumerate(instances):
        assert isinstance(instance, pat.NaTType), f"Instance {i} is not NaTType"
        assert instance is pd.NaT, f"Instance {i} is not the singleton pd.NaT (got {instance} vs {pd.NaT})"
```

**Failing input**: `n=1` (or any value)

## Reproducing the Bug

```python
import pandas as pd
import pandas.api.typing as pat

nat1 = pat.NaTType()
nat2 = pat.NaTType()
na1 = pat.NAType()
na2 = pat.NAType()

print(f"NAType() is singleton: {na1 is na2}")
print(f"NAType() returns pd.NA: {na1 is pd.NA}")

print(f"NaTType() is singleton: {nat1 is nat2}")
print(f"NaTType() returns pd.NaT: {nat1 is pd.NaT}")
```

Output:
```
NAType() is singleton: True
NAType() returns pd.NA: True
NaTType() is singleton: False
NaTType() returns pd.NaT: False
```

## Why This Is A Bug

The `pandas.api.typing` module exports both `NaTType` and `NAType` for type-hinting purposes. These types represent special sentinel values (`pd.NaT` and `pd.NA` respectively).

While `NAType()` correctly returns the singleton `pd.NA`, `NaTType()` creates new instances that are neither equal to nor identical to `pd.NaT` or each other. This API inconsistency violates user expectations and creates confusion.

Expected: `pat.NaTType()` should return `pd.NaT` (like `pat.NAType()` returns `pd.NA`)
Actual: `pat.NaTType()` creates distinct new instances

## Fix

The fix depends on whether NaTType is implemented in Cython or Python. The behavior should mirror NAType's singleton pattern. Without access to the Cython source, a high-level fix would be:

Implement a `__new__` method in NaTType that returns the existing `pd.NaT` singleton instead of creating new instances, matching the behavior of NAType.