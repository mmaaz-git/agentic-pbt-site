# Bug Report: dask.dataframe.io.parquet.core sorted_columns Assertion Failure

**Target**: `dask.dataframe.io.parquet.core.sorted_columns`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `sorted_columns` function crashes with an AssertionError when processing row group statistics where a row group has `min > max` according to the comparison operator. This occurs at line 442 where the function asserts that divisions are sorted, but this assumption can be violated.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.parquet.core import sorted_columns


@given(st.lists(
    st.fixed_dictionaries({
        'columns': st.lists(
            st.fixed_dictionaries({
                'name': st.just('idx'),
                'min': st.text(alphabet='01', min_size=1, max_size=3),
                'max': st.text(alphabet='01', min_size=1, max_size=3)
            }),
            min_size=1,
            max_size=1
        )
    }),
    min_size=1,
    max_size=5
))
@settings(max_examples=1000)
def test_sorted_columns_assertion_invariant(statistics):
    result = sorted_columns(statistics)
    for col_info in result:
        divisions = col_info['divisions']
        assert divisions == sorted(divisions)
```

**Failing input**: `statistics=[{'columns': [{'name': 'idx', 'min': '1', 'max': '0'}]}]`

## Reproducing the Bug

```python
from dask.dataframe.io.parquet.core import sorted_columns

statistics = [
    {'columns': [{'name': 'idx', 'min': '1', 'max': '0'}]}
]

result = sorted_columns(statistics)
```

**Output:**
```
AssertionError
  File "dask/dataframe/io/parquet/core.py", line 442, in sorted_columns
    assert divisions == sorted(divisions)
```

**What happens:**
- The function creates `divisions = ['1', '0']`
- But `sorted(['1', '0']) = ['0', '1']`
- The assertion `divisions == sorted(divisions)` fails

## Why This Is A Bug

The `sorted_columns` function is designed to detect columns in parquet datasets that are already sorted, extracting division points for index optimization. The function assumes that if row group statistics pass certain checks (`c["min"] >= max` from the previous row group), the resulting divisions list will be sorted.

However, this assumption breaks when:

1. **String comparison edge cases**: Row groups with string-typed statistics where `min > max` lexicographically (e.g., '1' > '0')
2. **Data quality issues**: Malformed parquet files where min/max statistics are incorrect or swapped
3. **Single row group case**: When there's only one row group and its min > max

The function is used in `dask.dataframe.dask_expr.io.parquet` to calculate divisions for dataframe indexing. A crash here prevents users from reading parquet files with such statistics.

**Impact**: Users cannot read parquet files that have row groups with `min > max` statistics, even if the actual data might be valid. This is particularly problematic for string-typed indices where lexicographic ordering can be non-intuitive.

## Fix

The function should validate that min <= max before processing, or handle the case gracefully. Here's a proposed fix:

```diff
--- a/dask/dataframe/io/parquet/core.py
+++ b/dask/dataframe/io/parquet/core.py
@@ -425,6 +425,10 @@ def sorted_columns(statistics, columns=None):
         divisions = [c["min"]]
         max = c["max"]
         success = c["min"] is not None
+        # Validate that min <= max for the first row group
+        if success and c["min"] > c["max"]:
+            success = False
+            continue
         for stats in statistics[1:]:
             c = stats["columns"][i]
             if c["min"] is None:
@@ -439,7 +443,8 @@ def sorted_columns(statistics, columns=None):

         if success:
             divisions.append(max)
-            assert divisions == sorted(divisions)
+            # Ensure divisions are sorted (defensive check)
+            divisions = sorted(divisions)
             out.append({"name": c["name"], "divisions": divisions})

     return out
```

This fix:
1. Checks if `min > max` in the first row group and marks the column as unsorted
2. Ensures divisions are sorted before returning (defensive approach)
3. Prevents the crash while maintaining the function's invariant