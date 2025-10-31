# Bug Report: RangeIndex._concat Ignores Name Parameter

**Target**: `pandas.core.indexes.range.RangeIndex._concat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `RangeIndex._concat` method fails to apply the `name` parameter when concatenating a single index, violating the API contract and creating inconsistent behavior with the base `Index` class.

## Property-Based Test

```python
from pandas import RangeIndex
from hypothesis import given, strategies as st, assume


@given(
    st.integers(min_value=-100, max_value=100),
    st.integers(min_value=-100, max_value=100),
    st.integers(min_value=1, max_value=20),
    st.text(min_size=1, max_size=20),
    st.text(min_size=1, max_size=20),
)
def test_concat_single_index_renames(start, stop, step, original_name, new_name):
    assume(start != stop)
    assume(original_name != new_name)

    idx = RangeIndex(start, stop, step, name=original_name)
    assume(len(idx) > 0)

    result = idx._concat([idx], name=new_name)

    assert result.name == new_name, f"_concat should rename to '{new_name}' but got '{result.name}'"
```

**Failing input**: `start=0, stop=1, step=1, original_name='0', new_name='6'`

## Reproducing the Bug

```python
from pandas import RangeIndex, Index

idx = RangeIndex(0, 3, name="original")
result = idx._concat([idx], name="new_name")

print(f"Expected name: 'new_name'")
print(f"Actual name: '{result.name}'")

base_idx = Index([0, 1, 2], name="original")
base_result = base_idx._concat([base_idx], name="new_name")
print(f"\nBase Index class name: '{base_result.name}'")
```

## Why This Is A Bug

The `_concat` method has a `name` parameter that should be applied to the resulting index. However, when `len(indexes) == 1`, the method returns `indexes[0]` directly without renaming it (line 950-951 in range.py):

```python
elif len(indexes) == 1:
    return indexes[0]
```

This violates the API contract because:
1. All other branches of the method correctly apply the name via `.rename(name)` (see line 993, 973, 984)
2. The base `Index` class correctly applies the name in all cases
3. Callers expect the `name` parameter to be honored regardless of input size

This creates inconsistent behavior where the same operation produces different names depending on how many indexes are concatenated.

## Fix

```diff
--- a/pandas/core/indexes/range.py
+++ b/pandas/core/indexes/range.py
@@ -948,7 +948,7 @@ class RangeIndex(Index):
             return super()._concat(indexes, name)

         elif len(indexes) == 1:
-            return indexes[0]
+            return indexes[0].rename(name)

         rng_indexes = cast(list[RangeIndex], indexes)
```