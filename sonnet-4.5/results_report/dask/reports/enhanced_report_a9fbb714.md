# Bug Report: dask.dataframe.dask_expr Repartition Division Count Invariant Violation

**Target**: `dask.dataframe.dask_expr._repartition.Repartition`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When repartitioning a DataFrame to increase the number of partitions, the `npartitions` property can report an incorrect value that doesn't match the actual number of partitions created. This violates the fundamental Dask DataFrame invariant `npartitions == len(divisions) - 1` and causes errors when trying to access non-existent partitions.

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

<details>

<summary>
**Failing input**: `n_rows=5, n_partitions_initial=2, n_partitions_target=4`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 28, in <module>
    test_repartition_division_count()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 8, in test_repartition_division_count
    n_rows=st.integers(min_value=5, max_value=1000),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 24, in test_repartition_division_count
    assert actual_division_count == expected_division_count, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Division count incorrect: expected 5, got 4
Falsifying example: test_repartition_division_count(
    n_rows=5,
    n_partitions_initial=2,
    n_partitions_target=4,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/28/hypo.py:25
```
</details>

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
print(f"Actual partitions: {repartitioned.npartitions}")

# Check the invariant
expected_partitions = len(repartitioned.divisions) - 1
print(f"\nInvariant check: npartitions ({repartitioned.npartitions}) == len(divisions) - 1 ({expected_partitions})?")
print(f"Invariant violated: {repartitioned.npartitions != expected_partitions}")
```

<details>

<summary>
Output showing invariant violation and incorrect npartitions value
</summary>
```
Original: npartitions=2, divisions=(0, 3, 4)
Requested: 4 partitions
Got: npartitions=4, divisions=(0, 1, 3, 4)
Division count: 4 (expected 5)
Actual partitions: 4

Invariant check: npartitions (4) == len(divisions) - 1 (3)?
Invariant violated: True
```
</details>

## Why This Is A Bug

This bug violates the fundamental Dask DataFrame invariant and causes actual runtime errors:

1. **Invariant Violation**: The fundamental Dask DataFrame invariant `npartitions == len(divisions) - 1` is violated. The `npartitions` property reports 4, but `len(divisions) - 1 = 3`.

2. **Incorrect Metadata**: The `npartitions` property returns 4 (the requested value) but only 3 partitions actually exist. This misleads users and downstream code about the DataFrame's actual structure.

3. **Runtime Errors**: Attempting to access partition 3 (which `npartitions` says exists) causes errors since only partitions 0-2 actually exist.

4. **Documentation Contradiction**: While the documentation states the partition count may be "slightly lower", it does NOT state that the `npartitions` property will report an incorrect value. Properties should report actual state, not requested state.

5. **Silent Corruption**: The bug occurs silently with no warnings, making it extremely difficult for users to debug when their computations fail.

## Relevant Context

The root cause is in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/dask_expr/_repartition.py`:

- Lines 111-131: The code uses `np.interp` to compute new division boundaries based on the requested partition count
- Line 121: For integer indices, divisions are cast to the original dtype, which can cause rounding
- Lines 130-131: The code removes duplicate divisions with `divisions = list(unique(divisions[:-1])) + [divisions[-1]]`
- Lines 61-71: The `npartitions` property returns the requested value (from `operand("new_partitions")`) rather than the actual partition count

When interpolating divisions for small DataFrames, the computed boundaries can round to the same integer values, creating duplicates. These duplicates are removed (line 131), reducing the actual partition count. However, the `npartitions` property still returns the originally requested value, creating the inconsistency.

Documentation link: https://docs.dask.org/en/latest/dataframe-api.html#dask.dataframe.DataFrame.repartition

## Proposed Fix

The `npartitions` property should return the actual number of partitions, not the requested number. The fix requires modifying the `Repartition` class to compute the actual partition count from divisions:

```diff
--- a/dask/dataframe/dask_expr/_repartition.py
+++ b/dask/dataframe/dask_expr/_repartition.py
@@ -61,14 +61,10 @@ class Repartition(Expr):

     @property
     def npartitions(self):
-        if (
-            "new_partitions" in self._parameters
-            and self.operand("new_partitions") is not None
-        ):
-            new_partitions = self.operand("new_partitions")
-            if isinstance(new_partitions, Callable):
-                return new_partitions(self.frame.npartitions)
-            return new_partitions
+        # Always return the actual number of partitions based on divisions
+        divs = self._divisions()
+        if divs is not None:
+            return len(divs) - 1
         return super().npartitions

     @functools.cached_property
```

Alternatively, if maintaining the requested value is intentional, the code should raise an error when unable to create the requested number of partitions rather than silently creating fewer partitions while reporting the wrong count.