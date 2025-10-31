# Bug Report: xarray.core.indexes.PandasIndex.roll crashes on empty index with ZeroDivisionError

**Target**: `xarray.core.indexes.PandasIndex.roll`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `PandasIndex.roll` method crashes with `ZeroDivisionError` when called on an empty index, instead of gracefully returning an empty rolled index.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import xarray.indexes as xr_indexes

@given(st.integers(min_value=-100, max_value=100))
def test_pandas_index_roll_on_empty_index(shift):
    """
    Property: roll should work on empty indexes without crashing.
    Rolling an empty index by any amount should return an empty index.
    """
    empty_pd_idx = pd.Index([])
    idx = xr_indexes.PandasIndex(empty_pd_idx, dim='x')

    result = idx.roll({'x': shift})

    assert len(result.index) == 0
    assert result.dim == idx.dim

# Run the test
if __name__ == "__main__":
    test_pandas_index_roll_on_empty_index()
```

<details>

<summary>
**Failing input**: `shift=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 21, in <module>
    test_pandas_index_roll_on_empty_index()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 6, in test_pandas_index_roll_on_empty_index
    def test_pandas_index_roll_on_empty_index(shift):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 14, in test_pandas_index_roll_on_empty_index
    result = idx.roll({'x': shift})
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/indexes.py", line 914, in roll
    shift = shifts[self.dim] % self.index.shape[0]
            ~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: integer modulo by zero
Falsifying example: test_pandas_index_roll_on_empty_index(
    shift=0,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import xarray.indexes as xr_indexes

# Create an empty pandas index
empty_pd_idx = pd.Index([])

# Create a PandasIndex object with the empty index
idx = xr_indexes.PandasIndex(empty_pd_idx, dim='x')

# Try to roll the empty index by 1 position
try:
    result = idx.roll({'x': 1})
    print("Roll succeeded. Result:", result)
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
```

<details>

<summary>
ZeroDivisionError: integer modulo by zero
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/repo.py", line 12, in <module>
    result = idx.roll({'x': 1})
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/indexes.py", line 914, in roll
    shift = shifts[self.dim] % self.index.shape[0]
            ~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: integer modulo by zero
Error type: ZeroDivisionError
Error message: integer modulo by zero

Full traceback:
```
</details>

## Why This Is A Bug

This violates expected behavior for several reasons:

1. **Empty indexes are valid objects**: The `PandasIndex` constructor accepts empty pandas.Index objects without error, making them valid inputs to all PandasIndex methods. The roll method should handle this valid input gracefully.

2. **Mathematical consistency**: Rolling an empty sequence by any amount should return an empty sequence. This is the behavior expected from similar operations across numpy, pandas, and other array libraries. For instance, `numpy.roll([], 1)` returns an empty array.

3. **Documentation contract violation**: The method docstring states it will "Roll this index by an offset along one or more dimensions" and return "A new index with rolled data." There's no mention that empty indexes are unsupported or will cause crashes.

4. **Real-world impact**: Empty indexes commonly occur in data processing pipelines after filtering operations that remove all elements. When this happens, the entire pipeline crashes instead of propagating the empty result forward.

5. **Implementation detail leaking**: The crash occurs due to an implementation detail (using modulo to wrap the shift value) rather than any fundamental limitation. The error message "integer modulo by zero" doesn't help users understand what went wrong with their roll operation.

## Relevant Context

The bug occurs at line 914 in `/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/indexes.py`:

```python
def roll(self, shifts: Mapping[Any, int]) -> PandasIndex:
    shift = shifts[self.dim] % self.index.shape[0]  # Crashes when shape[0] is 0
```

When `self.index.shape[0]` is 0 (empty index), the modulo operation causes division by zero. This is purely an implementation oversight - the modulo is used to normalize the shift value to be within the index bounds, but this normalization is unnecessary for empty indexes.

Related xarray documentation: https://docs.xarray.dev/en/stable/generated/xarray.indexes.PandasIndex.html

The PandasIndex.roll method inherits from the base Index.roll interface which documents that implementations are optional but doesn't exclude empty indexes as a special case.

## Proposed Fix

```diff
--- a/xarray/core/indexes.py
+++ b/xarray/core/indexes.py
@@ -911,7 +911,11 @@ class PandasIndex(Index):
         return {self.dim: get_indexer_nd(self.index, other.index, method, tolerance)}

     def roll(self, shifts: Mapping[Any, int]) -> PandasIndex:
-        shift = shifts[self.dim] % self.index.shape[0]
+        if self.index.shape[0] == 0:
+            # Empty index: rolling has no effect
+            return self._replace(self.index[:])
+
+        shift = shifts[self.dim] % self.index.shape[0]

         if shift != 0:
             new_pd_idx = self.index[-shift:].append(self.index[:-shift])
```