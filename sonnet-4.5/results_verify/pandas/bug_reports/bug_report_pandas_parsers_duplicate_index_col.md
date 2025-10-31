# Bug Report: pandas.io.parsers Duplicate index_col ValueError

**Target**: `pandas.io.parsers.base_parser._clean_index_names`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `read_csv()` is called with duplicate integer values in the `index_col` parameter (e.g., `index_col=[0, 0]`), it raises a `ValueError` due to attempting to remove the same column name from the columns list multiple times.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import pandas as pd
from io import StringIO

@settings(max_examples=100)
@given(
    ncols=st.integers(min_value=2, max_value=10),
    nrows=st.integers(min_value=1, max_value=20),
    duplicate_index=st.integers(min_value=0, max_value=5)
)
def test_duplicate_index_col_no_crash(ncols, nrows, duplicate_index):
    assume(duplicate_index < ncols)

    data = [[i * ncols + j for j in range(ncols)] for i in range(nrows)]
    df = pd.DataFrame(data, columns=[f"col{i}" for i in range(ncols)])
    csv_data = df.to_csv(index=False)

    pd.read_csv(StringIO(csv_data), index_col=[duplicate_index, duplicate_index])
```

**Failing input**: `index_col=[0, 0]` with any valid CSV data

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

csv_data = "a,b,c\n1,2,3\n4,5,6"

df = pd.read_csv(StringIO(csv_data), index_col=[0, 0])
```

**Expected error**:
```
ValueError: list.remove(x): x not in list
```

## Why This Is A Bug

The `_clean_index_names` function in `pandas/io/parsers/base_parser.py` (lines 1046-1081) removes column names from the `columns` list without checking for duplicates in `index_col`:

```python
for i, c in enumerate(index_col):
    if isinstance(c, str):
        # ... handle string case
    else:
        name = cp_cols[c]
        columns.remove(name)  # BUG: No check if already removed
        index_names.append(name)
```

When `index_col=[0, 0]`:
1. First iteration: `name = 'a'`, `columns.remove('a')` succeeds → `columns = ['b', 'c']`
2. Second iteration: `name = 'a'`, `columns.remove('a')` fails → ValueError

While duplicate indices in `index_col` may be unusual input, the function should either:
1. Silently handle duplicates (e.g., by using only unique values)
2. Raise a clear, informative error message explaining the issue
3. Document that duplicates are not allowed

Currently, it raises a confusing internal error that doesn't explain the root cause.

## Fix

```diff
--- a/pandas/io/parsers/base_parser.py
+++ b/pandas/io/parsers/base_parser.py
@@ -1060,13 +1060,20 @@ class ParserBase:
         # don't mutate
         index_col = list(index_col)

+        # Check for duplicates in index_col
+        if len(index_col) != len(set(index_col)):
+            raise ValueError(
+                "index_col must not contain duplicate column references. "
+                f"Got: {index_col}"
+            )
+
         for i, c in enumerate(index_col):
             if isinstance(c, str):
                 index_names.append(c)
                 for j, name in enumerate(cp_cols):
                     if name == c:
                         index_col[i] = j
                         columns.remove(name)
                         break
             else:
                 name = cp_cols[c]
```

This fix validates `index_col` early and provides a clear error message explaining the problem to users.