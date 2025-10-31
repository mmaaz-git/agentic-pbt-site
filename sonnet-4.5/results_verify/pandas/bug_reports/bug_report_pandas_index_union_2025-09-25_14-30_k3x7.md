# Bug Report: pandas Index.union Inconsistent Duplicate Handling

**Target**: `pandas.core.indexes.base.Index.union`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Index.union()` method inconsistently preserves duplicates, while other set operations (`intersection`, `difference`, `symmetric_difference`) correctly remove duplicates. This violates mathematical set theory principles and creates unexpected behavior for users.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from hypothesis import given, strategies as st, settings


@st.composite
def pandas_index_strategy(draw, elements=None):
    if elements is None:
        elements = st.one_of(
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(min_size=1, max_size=10)
        )
    values = draw(st.lists(elements, min_size=0, max_size=50))
    return pd.Index(values)


@given(idx=pandas_index_strategy())
@settings(max_examples=200)
def test_union_with_self_should_be_unique(idx):
    result = idx.union(idx)
    unique_idx = idx.drop_duplicates()
    assert result.equals(unique_idx), f"Union with self not unique: {result} != {unique_idx}"
```

**Failing input**: `idx=Index([0, 0], dtype='int64')`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd

idx1 = pd.Index([1, 1, 2])
idx2 = pd.Index([2, 2, 3])

print(f"idx1 = {idx1}")
print(f"idx2 = {idx2}")
print()

print(f"union:                {idx1.union(idx2)}")
print(f"  is_unique: {idx1.union(idx2).is_unique}")
print()

print(f"intersection:         {idx1.intersection(idx2)}")
print(f"  is_unique: {idx1.intersection(idx2).is_unique}")
print()

print(f"difference:           {idx1.difference(idx2)}")
print(f"  is_unique: {idx1.difference(idx2).is_unique}")
print()

print(f"symmetric_difference: {idx1.symmetric_difference(idx2)}")
print(f"  is_unique: {idx1.symmetric_difference(idx2).is_unique}")
```

**Output:**
```
idx1 = Index([1, 1, 2], dtype='int64')
idx2 = Index([2, 2, 3], dtype='int64')

union:                Index([1, 1, 2, 2, 3], dtype='int64')
  is_unique: False

intersection:         Index([2], dtype='int64')
  is_unique: True

difference:           Index([1], dtype='int64')
  is_unique: True

symmetric_difference: Index([1, 3], dtype='int64')
  is_unique: True
```

## Why This Is A Bug

1. **Mathematical inconsistency**: In set theory, union is a set operation that returns unique elements, just like intersection, difference, and symmetric difference. The current implementation violates this principle.

2. **Inconsistent behavior**: Three of the four set operations (`intersection`, `difference`, `symmetric_difference`) correctly return unique results, but `union` preserves duplicates. This inconsistency is confusing and error-prone.

3. **Violates user expectations**: Users familiar with set theory or Python's built-in `set` type expect `union` to return unique elements. The current behavior is surprising and undocumented.

4. **Internal acknowledgment**: The source code in `pandas/core/indexes/base.py` line 40-41 shows that `intersection` explicitly calls `.unique()` when inputs have duplicates:
   ```python
   if not self.is_unique:
       result = self.unique()._get_reconciled_name_object(other)
   ```
   However, the `_union` method at line 47-50 intentionally preserves duplicates:
   ```python
   elif not other.is_unique:
       result_dups = algos.union_with_duplicates(self, other)
       return _maybe_try_sort(result_dups, sort)
   ```

## Fix

The `_union` method should ensure unique results, consistent with other set operations. Here's a high-level approach:

1. In the `Index._union` method (around line 47-50 in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/base.py`), after computing the union, call `.unique()` on the result before returning.

2. Update the fast path at line 105-111 to also return unique results when `self.equals(other)` but has duplicates.

3. Update any tests that expect duplicate preservation from `union` to match the corrected behavior.

A more detailed fix would require modifying the `algos.union_with_duplicates` function to return unique results, or replacing it with a different algorithm that naturally produces unique outputs.