# Bug Report: dask.dataframe.dask_expr Series Quantile Mutates Input Array

**Target**: `dask.dataframe.dask_expr._quantile.SeriesQuantile.q`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `SeriesQuantile.q` cached property can mutate the user's input array when `q` is provided as a numpy array. This happens because `np.array()` does not always create a copy (especially in older NumPy versions), and the subsequent in-place sort modifies the original array.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import pandas as pd
import dask.dataframe as dd

@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
             min_size=2, max_size=10)
)
def test_quantile_does_not_mutate_input(q_list):
    df = pd.DataFrame({'x': range(100)})
    ddf = dd.from_pandas(df, npartitions=4)

    q_array = np.array(q_list)
    q_copy = q_array.copy()

    _ = ddf.x.quantile(q_array)

    assert np.array_equal(q_array, q_copy), \
        "Input array was mutated by quantile operation"
```

**Failing input**: Any numpy array, e.g., `q_array = np.array([0.9, 0.5, 0.1])`

## Reproducing the Bug

```python
import numpy as np
import pandas as pd
import dask.dataframe as dd

df = pd.DataFrame({'x': list(range(100))})
ddf = dd.from_pandas(df, npartitions=4)

q_original = np.array([0.9, 0.5, 0.1])
q_copy = q_original.copy()

print(f"Before: q_original = {q_original}")

result = ddf.x.quantile(q_original)

print(f"After:  q_original = {q_original}")
print(f"Expected:          {q_copy}")

if not np.array_equal(q_original, q_copy):
    print("\n*** BUG: Input array was mutated! ***")
    print(f"Original was: {q_copy}")
    print(f"Now is:       {q_original}")
```

**Expected output:**
```
Before: q_original = [0.9 0.5 0.1]
After:  q_original = [0.1 0.5 0.9]  # SORTED IN-PLACE!
Expected:          [0.9 0.5 0.1]

*** BUG: Input array was mutated! ***
Original was: [0.9 0.5 0.1]
Now is:       [0.1 0.5 0.9]
```

## Why This Is A Bug

The code at `_quantile.py:22-27` attempts to create a numpy array from the input, but `np.array()` does not guarantee a copy:

```python
@functools.cached_property
def q(self):
    q = np.array(self.operand("q"))  # May not create a copy!
    if q.ndim > 0:
        assert len(q) > 0, f"must provide non-empty q={q}"
        q.sort(kind="mergesort")  # Sorts IN-PLACE
        return q
    return np.asarray([self.operand("q")])
```

When the input is already a numpy array with a compatible dtype, `np.array()` may return a view rather than a copy (especially in NumPy < 2.0). The subsequent in-place sort then modifies the user's original array, which violates the principle of least surprise and can cause subtle bugs in user code.

## Fix

Explicitly request a copy when creating the array:

```diff
--- a/dask/dataframe/dask_expr/_quantile.py
+++ b/dask/dataframe/dask_expr/_quantile.py
@@ -19,7 +19,10 @@ class SeriesQuantile(Expr):

     @functools.cached_property
     def q(self):
-        q = np.array(self.operand("q"))
+        # Explicitly copy to avoid mutating user's input array
+        # In NumPy >= 2.0: use copy=True parameter
+        # For older NumPy: use np.array(...).copy()
+        q = np.array(self.operand("q"), copy=True)
         if q.ndim > 0:
             assert len(q) > 0, f"must provide non-empty q={q}"
             q.sort(kind="mergesort")
```

For NumPy < 2.0 compatibility (where `copy` parameter doesn't exist), use:

```diff
--- a/dask/dataframe/dask_expr/_quantile.py
+++ b/dask/dataframe/dask_expr/_quantile.py
@@ -19,7 +19,8 @@ class SeriesQuantile(Expr):

     @functools.cached_property
     def q(self):
-        q = np.array(self.operand("q"))
+        # Explicitly copy to avoid mutating user's input array
+        q = np.array(self.operand("q")).copy()
         if q.ndim > 0:
             assert len(q) > 0, f"must provide non-empty q={q}"
             q.sort(kind="mergesort")
```