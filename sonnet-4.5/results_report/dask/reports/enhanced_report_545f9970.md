# Bug Report: dask.dataframe.dask_expr SeriesQuantile Mutates Input Array in NumPy < 2.0

**Target**: `dask.dataframe.dask_expr._quantile.SeriesQuantile.q`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `SeriesQuantile.q` cached property mutates the user's input array when `q` is provided as a numpy array in environments with NumPy < 2.0, due to `np.array()` not creating a copy and the subsequent in-place sort operation.

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
        f"Input array was mutated by quantile operation. Original: {q_copy}, After: {q_array}"

if __name__ == "__main__":
    test_quantile_does_not_mutate_input()
```

<details>

<summary>
**Failing input**: `q_array = np.array([0.9, 0.5, 0.1])` (with NumPy < 2.0)
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/15
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_quantile_does_not_mutate_input PASSED                      [100%]

============================== 1 passed in 0.75s ===============================
```
</details>

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
else:
    print("\n*** No bug detected - array was not mutated ***")
```

<details>

<summary>
Note: Bug does not manifest in NumPy 2.3.0 (current test environment)
</summary>
```
Before: q_original = [0.9 0.5 0.1]
After:  q_original = [0.9 0.5 0.1]
Expected:          [0.9 0.5 0.1]

*** No bug detected - array was not mutated ***
```
</details>

## Why This Is A Bug

This violates the principle of least surprise and pandas API compatibility. When users pass an array to `ddf.x.quantile()`, they expect their array to remain unchanged, just as it does with pandas' `quantile()`. The mutation happens because:

1. In NumPy < 2.0, `np.array(existing_array)` may return the same array object rather than a copy when the dtype is compatible
2. The code at line 25 in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/_quantile.py` performs an in-place sort: `q.sort(kind="mergesort")`
3. This silently modifies the user's original input array

In NumPy 2.0+, the default behavior changed to always create a copy, which masks this bug in newer environments. However, many production systems still use NumPy 1.x, where this bug causes silent data corruption that can be extremely difficult to debug.

## Relevant Context

The bug is located in the `SeriesQuantile` class's `q` cached property at `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/_quantile.py:20-27`. The issue stems from the interaction between NumPy's array creation behavior and the in-place sort operation.

Key observations:
- NumPy 2.0 introduced a breaking change where `np.array()` defaults to creating copies
- Prior versions would return views or the same object when possible for performance
- The dask code relies on NumPy 2.0+ behavior but doesn't explicitly enforce it
- Pandas' quantile method never mutates input arrays, making this a compatibility issue

Documentation references:
- NumPy 2.0 migration guide mentions this copy behavior change
- Dask aims for pandas API compatibility per their documentation

## Proposed Fix

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