# Bug Report: xarray.core.indexes.PandasIndex.roll() ZeroDivisionError on Empty Index

**Target**: `xarray.core.indexes.PandasIndex.roll()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `PandasIndex.roll()` method crashes with a `ZeroDivisionError` when called on an empty index because it performs modulo operation by the index length without checking if the length is zero.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from xarray.core.indexes import PandasIndex

@st.composite
def xarray_pandas_indexes_including_empty(draw):
    size = draw(st.integers(min_value=0, max_value=100))
    if size == 0:
        pd_index = pd.Index([])
    else:
        values = draw(st.lists(st.integers(), min_size=size, max_size=size))
        pd_index = pd.Index(values)
    dim_name = draw(st.text(min_size=1, max_size=10))
    return PandasIndex(pd_index, dim_name)

@settings(max_examples=200)
@given(xarray_pandas_indexes_including_empty(), st.integers(min_value=-100, max_value=100))
def test_pandasindex_roll_no_crash(index, shift):
    dim = index.dim
    rolled = index.roll({dim: shift})
    # The test passes if no exception is raised

if __name__ == "__main__":
    test_pandasindex_roll_no_crash()
```

<details>

<summary>
**Failing input**: `PandasIndex(Index([], dtype='object', name='0'))` with shift=0
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 24, in <module>
    test_pandasindex_roll_no_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 17, in test_pandasindex_roll_no_crash
    @given(xarray_pandas_indexes_including_empty(), st.integers(min_value=-100, max_value=100))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 20, in test_pandasindex_roll_no_crash
    rolled = index.roll({dim: shift})
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/indexes.py", line 914, in roll
    shift = shifts[self.dim] % self.index.shape[0]
            ~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: integer modulo by zero
Falsifying example: test_pandasindex_roll_no_crash(
    index=PandasIndex(Index([], dtype='object', name='0')),
    shift=0,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/58/hypo.py:9
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py:1193
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/base.py:559
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from xarray.core.indexes import PandasIndex

# Create an empty pandas index
empty_idx = pd.Index([])

# Create a PandasIndex with the empty index
xr_idx = PandasIndex(empty_idx, "x")

print(f"Index length: {len(xr_idx.index)}")
print(f"Index shape: {xr_idx.index.shape}")
print(f"Attempting to roll by 1...")

# Attempt to roll the empty index - this will crash with ZeroDivisionError
rolled = xr_idx.roll({"x": 1})
print("Roll succeeded!")
```

<details>

<summary>
ZeroDivisionError when rolling empty PandasIndex
</summary>
```
Index length: 0
Index shape: (0,)
Attempting to roll by 1...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/repo.py", line 15, in <module>
    rolled = xr_idx.roll({"x": 1})
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/indexes.py", line 914, in roll
    shift = shifts[self.dim] % self.index.shape[0]
            ~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: integer modulo by zero
```
</details>

## Why This Is A Bug

This violates expected behavior for several reasons:

1. **Mathematical consistency**: Rolling an empty collection is a well-defined operation - it should return the empty collection unchanged. There are no elements to rotate, so the result is trivially empty.

2. **NumPy compatibility**: NumPy's `np.roll()` handles empty arrays gracefully, returning an empty array. Since xarray builds on NumPy conventions, users expect similar behavior:
   ```python
   np.roll(np.array([]), 1)  # Returns array([])
   ```

3. **DataArray documentation**: The DataArray.roll() documentation states that roll "treats the given dimensions as periodic" and "will not create any missing values." This implies the operation should be valid on any sized array, including empty ones.

4. **Unhelpful error message**: The `ZeroDivisionError` doesn't indicate the actual problem (empty index). Users encountering this error won't immediately understand that the issue is with empty indexes unless they examine the stack trace carefully.

5. **Common in data processing**: Empty collections frequently appear in data processing pipelines (filtered data, edge cases, initialization). Operations should handle these gracefully rather than crashing.

## Relevant Context

The bug occurs in `/xarray/core/indexes.py` at line 914:
```python
def roll(self, shifts: Mapping[Any, int]) -> PandasIndex:
    shift = shifts[self.dim] % self.index.shape[0]  # ZeroDivisionError when shape[0] == 0
```

The code attempts to normalize the shift amount using modulo to handle shifts larger than the index size, but doesn't account for the case where the index has zero elements.

Related xarray documentation: https://docs.xarray.dev/en/stable/generated/xarray.DataArray.roll.html

The base Index class documentation indicates that roll() can be "re-implemented in subclasses of Index," suggesting each implementation should handle its edge cases appropriately.

## Proposed Fix

```diff
--- a/xarray/core/indexes.py
+++ b/xarray/core/indexes.py
@@ -911,6 +911,11 @@ class PandasIndex(Index):
         return {self.dim: get_indexer_nd(self.index, other.index, method, tolerance)}

     def roll(self, shifts: Mapping[Any, int]) -> PandasIndex:
+        # Handle empty index case - nothing to roll
+        if self.index.shape[0] == 0:
+            # Return a copy of the empty index unchanged
+            return self._replace(self.index[:])
+
         shift = shifts[self.dim] % self.index.shape[0]

         if shift != 0:
```