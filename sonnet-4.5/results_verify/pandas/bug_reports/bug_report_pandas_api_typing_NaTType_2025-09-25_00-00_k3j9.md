# Bug Report: pandas.api.typing.NaTType Constructor Not Singleton

**Target**: `pandas.api.typing.NaTType`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`NaTType()` constructor creates new instances on each call, inconsistent with `NAType()` which returns a singleton. This violates API consistency and user expectations.

## Property-Based Test

```python
import pandas.api.typing as pat
from hypothesis import given, strategies as st, settings


@settings(max_examples=100)
@given(st.integers(min_value=0, max_value=1000))
def test_nattype_singleton_property(n):
    instances = [pat.NaTType() for _ in range(n) if n > 0]
    if instances:
        first = instances[0]
        for instance in instances[1:]:
            assert instance is first, f"NaTType() should be a singleton (like NAType), but got different instances"
```

**Failing input**: `n=2`

## Reproducing the Bug

```python
import pandas.api.typing as pat

na1 = pat.NAType()
na2 = pat.NAType()
print(f"NAType() is singleton: {na1 is na2}")

nat1 = pat.NaTType()
nat2 = pat.NaTType()
print(f"NaTType() is singleton: {nat1 is nat2}")
```

Output:
```
NAType() is singleton: True
NaTType() is singleton: False
```

## Why This Is A Bug

The `pandas.api.typing` module exports both `NAType` and `NaTType` as parallel missing-value indicators. `NAType` is explicitly documented as "The NA singleton" and calling `NAType()` correctly returns the singleton instance. However, `NaTType()` creates new instances on each call, which is inconsistent.

While users should use `pd.NaT` directly rather than calling the constructor, the API inconsistency is confusing and violates the principle of least surprise. Both constructors should behave the same way.

Additionally, this causes a subtle hash-equality contract issue: multiple `NaTType()` instances have identical hashes but are not equal (due to NaN-like semantics) and not identical, which is unusual.

## Fix

The fix would require implementing a `__new__` method in the C extension that returns the singleton instance, similar to how `NAType` is implemented. Without access to the C source, a high-level approach would be:

1. Ensure `NaTType.__new__` returns the singleton `pd.NaT` instance
2. Make `NaTType()` behave consistently with `NAType()`