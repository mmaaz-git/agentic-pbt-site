# Bug Report: dask.dataframe.io.parquet _aggregate_columns Infinite Loop

**Target**: `dask.dataframe.dask_expr.io.parquet._aggregate_columns`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_aggregate_columns` function enters an infinite loop when called with an empty list as input due to a logic error in the loop termination condition.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout - likely infinite loop")

@settings(max_examples=100, deadline=2000)
@given(
    st.lists(
        st.lists(st.dictionaries(st.text(), st.integers())),
        min_size=0,
        max_size=5
    )
)
def test_aggregate_columns_terminates(cols):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(1)
    try:
        _aggregate_columns(cols, {})
        signal.alarm(0)
    except TimeoutException:
        signal.alarm(0)
        assert False, f"Function hung with input: {cols}"
```

**Failing input**: `[]`

## Reproducing the Bug

The bug can be demonstrated through static code analysis of the function at lines 1935-1948:

```python
def _aggregate_columns(cols, agg_cols):
    combine = []
    i = 0
    while True:
        inner = []
        combine.append(inner)
        try:
            for col in cols:
                inner.append(col[i])
        except IndexError:
            combine.pop()
            break
        i += 1
    return [_agg_dicts(c, agg_cols) for c in combine]
```

When `cols = []`:
1. The `while True` loop starts with `i = 0`
2. A new empty `inner` list is created and appended to `combine`
3. The `for col in cols` loop executes zero iterations (since `cols` is empty)
4. No `IndexError` is raised because the loop body never executes
5. `i` is incremented to 1
6. The loop continues indefinitely, creating empty lists and incrementing `i`

## Why This Is A Bug

The function has a logic error in its termination condition. The code assumes that iterating through `cols` will eventually raise an `IndexError`, but when `cols` is empty, the for loop never executes at all, so no `IndexError` can be raised. This violates the expected behavior that all functions should terminate in finite time for all valid inputs.

While this bug may not be reachable through normal usage (as `_agg_dicts` typically doesn't pass empty lists to aggregation functions), it represents a latent defect that could cause production hangs if:
- The function is called directly in tests or other code
- Future code changes create a path where an empty list is passed
- The function is reused in a different context

## Fix

```diff
--- a/dask/dataframe/dask_expr/io/parquet.py
+++ b/dask/dataframe/dask_expr/io/parquet.py
@@ -1933,6 +1933,9 @@ def _aggregate_columns(cols, agg_cols):
     """

 def _aggregate_columns(cols, agg_cols):
+    if not cols:
+        return []
+
     combine = []
     i = 0
     while True:
```

This fix adds an early return for the empty list case, ensuring the function terminates correctly for all inputs.