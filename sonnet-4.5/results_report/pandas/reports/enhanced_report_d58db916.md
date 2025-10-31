# Bug Report: pandas.api.typing.NaTType Singleton Pattern Violation

**Target**: `pandas.api.typing.NaTType`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `NaTType()` constructor creates distinct instances on each call instead of returning the singleton `pd.NaT`, violating the expected singleton pattern and creating inconsistency with `NAType`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.api.typing as typing
import pandas as pd


@given(st.integers(min_value=0, max_value=100))
@settings(max_examples=100)
def test_nattype_singleton_property(n):
    instances = [typing.NaTType() for _ in range(n)]
    if len(instances) > 0:
        first = instances[0]
        for instance in instances[1:]:
            assert instance is first, f"NaTType() should always return the same singleton instance"
            assert instance is pd.NaT, f"NaTType() should return pd.NaT"

if __name__ == "__main__":
    test_nattype_singleton_property()
```

<details>

<summary>
**Failing input**: `n=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 17, in <module>
    test_nattype_singleton_property()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 7, in test_nattype_singleton_property
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 13, in test_nattype_singleton_property
    assert instance is first, f"NaTType() should always return the same singleton instance"
           ^^^^^^^^^^^^^^^^^
AssertionError: NaTType() should always return the same singleton instance
Falsifying example: test_nattype_singleton_property(
    n=2,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import pandas.api.typing as typing

# Test NaTType singleton behavior
nat1 = typing.NaTType()
nat2 = typing.NaTType()

print(f"nat1 is nat2: {nat1 is nat2}")
print(f"nat1 is pd.NaT: {nat1 is pd.NaT}")
print(f"nat1 == nat2: {nat1 == nat2}")
print(f"nat1 == pd.NaT: {nat1 == pd.NaT}")

# Test NAType singleton behavior for comparison
na1 = typing.NAType()
na2 = typing.NAType()

print(f"\nna1 is na2: {na1 is na2}")
print(f"na1 is pd.NA: {na1 is pd.NA}")
print(f"na1 == na2: {na1 == na2}")
print(f"na1 == pd.NA: {na1 == pd.NA}")

# Additional diagnostics
print(f"\nType of nat1: {type(nat1)}")
print(f"Type of pd.NaT: {type(pd.NaT)}")
print(f"id(nat1): {id(nat1)}")
print(f"id(nat2): {id(nat2)}")
print(f"id(pd.NaT): {id(pd.NaT)}")
```

<details>

<summary>
NaTType creates new instances instead of returning singleton
</summary>
```
nat1 is nat2: False
nat1 is pd.NaT: False
nat1 == nat2: False
nat1 == pd.NaT: False

na1 is na2: True
na1 is pd.NA: True
na1 == na2: <NA>
na1 == pd.NA: <NA>

Type of nat1: <class 'pandas._libs.tslibs.nattype.NaTType'>
Type of pd.NaT: <class 'pandas._libs.tslibs.nattype.NaTType'>
id(nat1): 133770094475216
id(nat2): 133770094475888
id(pd.NaT): 133767264878224
```
</details>

## Why This Is A Bug

This bug violates the documented behavior and expected design patterns of pandas' missing value types:

1. **Singleton pattern violation**: NaT (Not-a-Time) is designed to be a singleton sentinel value representing missing timestamps, similar to `None` in Python. The pandas documentation refers to `pd.NaT` as "the missing value for DatetimeIndex, TimedeltaIndex." Creating multiple distinct instances breaks this fundamental design principle.

2. **Inconsistency with NAType**: The companion class `NAType()` correctly returns the singleton `pd.NA` instance on every call, establishing a clear pattern that users would reasonably expect `NaTType()` to follow. This inconsistency creates confusion and unexpected behavior.

3. **Identity check failures**: Much pandas code and user code relies on identity checks (`is pd.NaT`) for performance and correctness. These checks fail when using instances created via `NaTType()`, potentially causing subtle bugs in downstream code.

4. **Equality semantics broken**: Not only do the created instances fail identity checks, they also don't compare equal to each other or to `pd.NaT` (all comparisons return `False`), completely breaking the expected singleton semantics where all references should be to the same object.

## Relevant Context

The `NaTType` class is implemented in Cython (located at `pandas/_libs/tslibs/nattype.cpython-*.so`), which explains why the `__new__` method isn't properly returning the singleton instance. The `NAType` class likely has special handling to ensure singleton behavior that `NaTType` is missing.

This bug affects pandas version 2.3.2 and likely other recent versions. The issue is in the `pandas.api.typing` module which is used for type hints and runtime type checking.

Documentation references:
- pandas.NaT documentation: https://pandas.pydata.org/docs/reference/api/pandas.NaT.html
- pandas.api.typing documentation: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.NaTType.html

## Proposed Fix

The fix requires modifying the Cython implementation of `NaTType.__new__` to return the singleton instance instead of creating new objects. The implementation should follow this pattern:

```diff
# In pandas/_libs/tslibs/nattype.pyx or similar Cython file
cdef class NaTType:
    def __new__(cls):
-       # Current implementation that creates new instance
-       return <create new NaTType instance>
+       # Return the singleton NaT instance
+       from pandas import NaT
+       return NaT
```

Alternatively, if the class needs to maintain compatibility, the implementation could cache and return a single instance:

```diff
cdef class NaTType:
+   cdef object _instance = None

    def __new__(cls):
-       return <create new NaTType instance>
+       if cls._instance is None:
+           cls._instance = <create NaTType instance>
+       return cls._instance
```

This ensures all calls to `NaTType()` return the same singleton instance, matching the behavior of `NAType` and maintaining consistency across pandas' missing value types.