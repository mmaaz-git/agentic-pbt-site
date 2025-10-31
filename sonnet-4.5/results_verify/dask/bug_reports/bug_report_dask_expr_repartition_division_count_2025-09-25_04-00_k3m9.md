# Bug Report: dask.dataframe.dask_expr Repartition Division Count Mismatch

**Target**: `dask.dataframe.dask_expr._repartition.Repartition`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When repartitioning a DataFrame to a specific number of partitions using `repartition(npartitions=N)`, the actual number of partitions created can be less than `N` when the DataFrame has few rows. This violates the API contract and the fundamental invariant that `npartitions == len(divisions) - 1`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import numpy as np
from dask.dataframe.dask_expr import from_pandas


@given(
    n_rows=st.integers(min_value=5, max_value=1000),
    n_partitions_initial=st.integers(min_value=2, max_value=20),
    n_partitions_target=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=200)
def test_repartition_division_count(n_rows, n_partitions_initial, n_partitions_target):
    pdf = pd.DataFrame({
        'x': np.arange(n_rows)
    }, index=np.arange(n_rows))

    df = from_pandas(pdf, npartitions=n_partitions_initial)
    repartitioned = df.repartition(npartitions=n_partitions_target)

    expected_division_count = n_partitions_target + 1
    actual_division_count = len(repartitioned.divisions)

    assert actual_division_count == expected_division_count, \
        f"Division count incorrect: expected {expected_division_count}, got {actual_division_count}"
```

**Failing input**: `n_rows=5, n_partitions_initial=2, n_partitions_target=4`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np
from dask.dataframe.dask_expr import from_pandas

pdf = pd.DataFrame({'x': np.arange(5)}, index=np.arange(5))
df = from_pandas(pdf, npartitions=2)

print(f"Original: npartitions={df.npartitions}, divisions={df.divisions}")

repartitioned = df.repartition(npartitions=4)

print(f"Requested: 4 partitions")
print(f"Got: npartitions={repartitioned.npartitions}, divisions={repartitioned.divisions}")
print(f"Division count: {len(repartitioned.divisions)} (expected 5)")
```

**Output:**
```
Original: npartitions=2, divisions=(0, 2, 4)
Requested: 4 partitions
Got: npartitions=3, divisions=(0, 1, 2, 4)
Division count: 4 (expected 5)
```

## Why This Is A Bug

1. **API Contract Violation**: When calling `repartition(npartitions=N)`, users expect to get exactly `N` partitions. The function name and parameter clearly indicate this intent.

2. **Invariant Violation**: The fundamental Dask DataFrame invariant `npartitions == len(divisions) - 1` should always hold, but this bug causes a mismatch between the requested `npartitions` and the actual number of divisions created.

3. **Silent Failure**: The function silently returns fewer partitions than requested without any warning or error, making it difficult for users to detect this issue.

4. **Root Cause**: In `_repartition.py` lines 130-131, the code removes duplicate divisions:
   ```python
   # Ensure the computed divisions are unique
   divisions = list(unique(divisions[:-1])) + [divisions[-1]]
   ```
   When interpolating division boundaries for small DataFrames, the computed boundaries can collapse to the same values (especially after integer conversion on line 121), resulting in duplicates that are then removed, reducing the partition count below the requested value.

## Fix

The repartition logic should either:

**Option 1**: Validate and raise an error when unable to create the requested number of partitions:

```diff
--- a/_repartition.py
+++ b/_repartition.py
@@ -128,7 +128,12 @@
                     divisions[-1] = df.divisions[-1]

                     # Ensure the computed divisions are unique
                     divisions = list(unique(divisions[:-1])) + [divisions[-1]]
+
+                    if len(divisions) - 1 != npartitions:
+                        raise ValueError(
+                            f"Cannot create {npartitions} partitions with only "
+                            f"{len(df.divisions) - 1} unique division points. "
+                            f"Try using fewer partitions or a larger dataset."
+                        )
                     return RepartitionDivisions(df, divisions, self.force)
```

**Option 2**: Use a different strategy that doesn't rely solely on interpolation for small datasets. For example, allow non-unique divisions when necessary, or use a hybrid approach that creates empty partitions to maintain the requested partition count.

The current behavior of silently returning fewer partitions than requested is problematic and should be fixed.