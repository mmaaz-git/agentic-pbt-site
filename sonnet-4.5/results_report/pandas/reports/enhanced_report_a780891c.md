# Bug Report: RangeIndex._shallow_copy Returns Index Instead of RangeIndex for Single-Element Ranges

**Target**: `pandas.core.indexes.range.RangeIndex._shallow_copy`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`RangeIndex._shallow_copy` fails to return a memory-efficient `RangeIndex` for single-element equally-spaced arrays, returning a regular `Index` instead, violating its documented optimization goal.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas import RangeIndex
import numpy as np

@given(
    st.integers(min_value=-1000, max_value=1000),
    st.integers(min_value=-1000, max_value=1000),
    st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0)
)
def test_rangeindex_shallow_copy_with_equally_spaced_values(start, stop, step):
    """RangeIndex._shallow_copy should return RangeIndex for equally spaced values."""
    ri = RangeIndex(start, stop, step)
    if len(ri) == 0:
        return

    values = np.array(list(ri))
    result = ri._shallow_copy(values)

    # BUG: Fails for single-element ranges
    assert isinstance(result, RangeIndex), \
        f"Expected RangeIndex for equally-spaced values, got {type(result)}"

# Run the test
if __name__ == "__main__":
    test_rangeindex_shallow_copy_with_equally_spaced_values()
```

<details>

<summary>
**Failing input**: `start=0, stop=1, step=1`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/22
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_rangeindex_shallow_copy_with_equally_spaced_values FAILED  [100%]

=================================== FAILURES ===================================
___________ test_rangeindex_shallow_copy_with_equally_spaced_values ____________

    @given(
>       st.integers(min_value=-1000, max_value=1000),
                   ^^^
        st.integers(min_value=-1000, max_value=1000),
        st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0)
    )

hypo.py:7:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

start = 0, stop = 1, step = 1

    @given(
        st.integers(min_value=-1000, max_value=1000),
        st.integers(min_value=-1000, max_value=1000),
        st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0)
    )
    def test_rangeindex_shallow_copy_with_equally_spaced_values(start, stop, step):
        """RangeIndex._shallow_copy should return RangeIndex for equally spaced values."""
        ri = RangeIndex(start, stop, step)
        if len(ri) == 0:
            return

        values = np.array(list(ri))
        result = ri._shallow_copy(values)

        # BUG: Fails for single-element ranges
>       assert isinstance(result, RangeIndex), \
            f"Expected RangeIndex for equally-spaced values, got {type(result)}"
E       AssertionError: Expected RangeIndex for equally-spaced values, got <class 'pandas.core.indexes.base.Index'>
E       assert False
E        +  where False = isinstance(Index([0], dtype='int64'), RangeIndex)
E       Falsifying example: test_rangeindex_shallow_copy_with_equally_spaced_values(
E           start=0,
E           stop=1,
E           step=1,
E       )
E       Explanation:
E           These lines were always and only run by failing examples:
E               /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/base.py:667
E               /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/base.py:1354
E               /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/range.py:481

hypo.py:21: AssertionError
=========================== short test summary info ============================
FAILED hypo.py::test_rangeindex_shallow_copy_with_equally_spaced_values - Ass...
============================== 1 failed in 0.51s ===============================
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from pandas import RangeIndex
import numpy as np

ri = RangeIndex(0, 1, 1)
values = np.array([0])

result = ri._shallow_copy(values)

print(f"Input values: {values}")
print(f"Result type: {type(result).__name__}")
print(f"Expected: RangeIndex (for equally-spaced values)")
print(f"Actual: {type(result).__name__}")
print(f"Is RangeIndex: {isinstance(result, RangeIndex)}")
```

<details>

<summary>
Output shows Index returned instead of RangeIndex
</summary>
```
Input values: [0]
Result type: Index
Expected: RangeIndex (for equally-spaced values)
Actual: Index
Is RangeIndex: False
```
</details>

## Why This Is A Bug

1. **Violates documented intent**: The code comment at lines 473-474 in `pandas/core/indexes/range.py` explicitly states:
   > "GH 46675 & 43885: If values is equally spaced, return a more memory-compact RangeIndex instead of Index with 64-bit dtype"

   A single-element array is mathematically equally-spaced (trivially so, as there are no consecutive pairs to have differing deltas).

2. **Inconsistent behavior**: The method correctly returns `RangeIndex` for arrays with 2+ equally-spaced elements but fails for single-element arrays, creating an unnecessary edge case.

3. **Memory inefficiency**: Returns a full `Index` object that stores all values in memory instead of the more compact `RangeIndex` that only stores start/stop/step parameters.

4. **Root cause**: The `unique_deltas` function (imported from `pandas._libs.algos`) returns an empty array for single-element inputs since it computes differences between consecutive elements. For a single element, there are no consecutive pairs, resulting in `len(unique_diffs) == 0` instead of the expected `1`, causing the optimization check to fail.

## Relevant Context

The `unique_deltas` function behavior:
- For empty arrays: returns empty array (length 0)
- For single-element arrays: returns empty array (length 0)
- For two+ element arrays: returns array of unique differences (length ≥ 1)

This edge case affects the optimization path at line 475-479 in `range.py`:
```python
unique_diffs = unique_deltas(values)
if len(unique_diffs) == 1 and unique_diffs[0] != 0:
    # This condition fails for single elements since len(unique_diffs) == 0
```

The issue was introduced when optimizing for memory efficiency in GitHub issues #46675 and #43885, but the single-element edge case wasn't considered.

## Proposed Fix

```diff
--- a/pandas/core/indexes/range.py
+++ b/pandas/core/indexes/range.py
@@ -470,6 +470,11 @@ class RangeIndex(Index):

     if values.dtype.kind == "f":
         return Index(values, name=name, dtype=np.float64)
+
+    # Handle single-element arrays: they are trivially equally-spaced
+    if len(values) == 1:
+        val = values[0]
+        return type(self)._simple_new(range(val, val + 1), name=name)
     # GH 46675 & 43885: If values is equally spaced, return a
     # more memory-compact RangeIndex instead of Index with 64-bit dtype
     unique_diffs = unique_deltas(values)
```