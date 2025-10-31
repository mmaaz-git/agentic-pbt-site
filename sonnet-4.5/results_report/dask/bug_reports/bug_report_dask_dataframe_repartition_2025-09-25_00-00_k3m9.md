# Bug Report: dask.dataframe.repartition AssertionError

**Target**: `dask.dataframe.DataFrame.repartition`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When attempting to repartition a Dask DataFrame with unknown divisions (created with `sort=False`) from 1 partition to more partitions when there is insufficient data, the operation fails with an `AssertionError` instead of either gracefully handling the case or raising an appropriate user-facing exception.

## Property-Based Test

```python
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import data_frames, column
import dask.dataframe as dd


@settings(max_examples=100)
@given(
    df=data_frames(
        columns=[
            column("a", elements=st.integers(-100, 100)),
            column("b", elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)),
        ],
    ),
    npartitions1=st.integers(min_value=1, max_value=5),
    npartitions2=st.integers(min_value=1, max_value=5),
)
def test_repartition_preserves_data(df, npartitions1, npartitions2):
    if len(df) == 0:
        npartitions1 = 1
        npartitions2 = 1

    ddf = dd.from_pandas(df, npartitions=npartitions1, sort=False)
    repartitioned = ddf.repartition(npartitions=npartitions2)
    result = repartitioned.compute()

    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True), check_dtype=False)
```

**Failing input**:
```python
df = pd.DataFrame({'a': [0], 'b': [0.0]})
npartitions1 = 1
npartitions2 = 2
```

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

df = pd.DataFrame({'a': [0], 'b': [0.0]})

ddf = dd.from_pandas(df, npartitions=1, sort=False)

repartitioned = ddf.repartition(npartitions=2)
result = repartitioned.compute()
```

**Output**:
```
AssertionError
```

**Full traceback**:
```
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_repartition.py", line 192, in _partitions_boundaries
    assert npartitions_input > npartitions
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

## Why This Is A Bug

1. **Incorrect error handling**: The code uses an `assert` statement for validation, which is inappropriate for user-facing error handling. Assertions can be disabled with Python's `-O` flag and should only be used for internal invariants, not for validating user inputs.

2. **Unexpected behavior**: The documentation for `repartition` states that "The number of partitions used may be slightly lower than npartitions depending on data distribution, but will never be higher." This suggests the operation should either succeed with fewer partitions or raise a clear ValueError, not crash with an AssertionError.

3. **Poor user experience**: Users attempting to repartition small DataFrames receive a cryptic AssertionError from internal code rather than a helpful error message explaining why the operation cannot proceed.

The root cause is in `/dask/dataframe/dask_expr/_repartition.py` in the `RepartitionToFewer` class, which has an assertion that `npartitions_input > npartitions`. However, when divisions are unknown and the DataFrame has insufficient data, the optimizer incorrectly routes the operation to `RepartitionToFewer` even when trying to increase partitions.

## Fix

```diff
--- a/dask/dataframe/dask_expr/_repartition.py
+++ b/dask/dataframe/dask_expr/_repartition.py
@@ -189,7 +189,11 @@ class RepartitionToFewer(Repartition):
     def _partitions_boundaries(self):
         npartitions = self.new_partitions
         npartitions_input = self.frame.npartitions
-        assert npartitions_input > npartitions
+        if npartitions_input <= npartitions:
+            raise ValueError(
+                f"Cannot repartition from {npartitions_input} to {npartitions} partitions. "
+                f"RepartitionToFewer requires fewer output partitions than input partitions."
+            )
         return self._compute_partition_boundaries(npartitions, npartitions_input)
```

However, a more complete fix would involve ensuring that the `Repartition._lower()` method correctly handles all edge cases and routes operations to the appropriate strategy. The real issue is that when a DataFrame with unknown divisions and insufficient data attempts to increase partitions, it incorrectly gets routed to `RepartitionToFewer`.