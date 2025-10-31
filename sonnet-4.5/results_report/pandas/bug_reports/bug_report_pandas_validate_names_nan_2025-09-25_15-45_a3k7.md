# Bug Report: pandas.io.parsers._validate_names NaN Duplicate Detection

**Target**: `pandas.io.parsers.readers._validate_names`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_validate_names` function fails to detect duplicate NaN values in column names due to NaN's non-reflexive equality property. The function uses `set()` for duplicate detection, but `NaN != NaN` causes multiple NaN values to be treated as distinct elements in the set.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.parsers.readers import _validate_names

@given(st.lists(st.floats(allow_nan=True), min_size=2, max_size=10))
def test_validate_names_detects_nan_duplicates(names):
    nan_count = sum(1 for x in names if isinstance(x, float) and math.isnan(x))
    if nan_count > 1:
        try:
            _validate_names(names)
            assert False, f"Should reject duplicate NaN in {names}"
        except ValueError:
            pass
```

**Failing input**: `[nan, nan]`

## Reproducing the Bug

```python
from pandas.io.parsers.readers import _validate_names

names = [float('nan'), float('nan')]
_validate_names(names)

print(f"Input: {names}")
print(f"len(names): {len(names)}")
print(f"len(set(names)): {len(set(names))}")
print(f"set(names): {set(names)}")
```

Output:
```
Input: [nan, nan]
len(names): 2
len(set(names)): 2
set(names): {nan, nan}
```

The function accepts the duplicate NaN values without raising ValueError.

## Why This Is A Bug

1. **Violates documented behavior**: The docstring at line 561 states "Raise ValueError if the `names` parameter contains duplicates", but duplicate NaN values are not detected.

2. **Root cause** (line 575 in `/pandas/io/parsers/readers.py`):
   ```python
   if len(names) != len(set(names)):
       raise ValueError("Duplicate names are not allowed.")
   ```
   This check fails for NaN because `set([nan, nan])` creates a set with 2 elements due to `NaN != NaN`.

3. **Inconsistent with user expectations**: Users expect that using the same value twice (even if it's NaN) should be detected as a duplicate.

4. **Real-world impact**: Pandas allows creating DataFrames with duplicate NaN column names:
   ```python
   import pandas as pd
   from io import StringIO

   csv = StringIO("1,2\n3,4")
   df = pd.read_csv(csv, names=[float('nan'), float('nan')])
   # This succeeds, creating a DataFrame with duplicate NaN column names
   ```

## Fix

```diff
--- a/pandas/io/parsers/readers.py
+++ b/pandas/io/parsers/readers.py
@@ -1,5 +1,6 @@
 from __future__ import annotations

+import math
 from collections import (
     abc,
     defaultdict,
@@ -572,7 +573,18 @@ def _validate_names(names: Sequence[Hashable] | None) -> None:
         If names are not unique or are not ordered (e.g. set).
     """
     if names is not None:
-        if len(names) != len(set(names)):
+        # Check for NaN duplicates separately since NaN != NaN
+        nan_count = 0
+        for name in names:
+            if isinstance(name, float) and math.isnan(name):
+                nan_count += 1
+                if nan_count > 1:
+                    raise ValueError("Duplicate names are not allowed.")
+
+        # Check for other duplicates using set
+        non_nan_names = [n for n in names if not (isinstance(n, float) and math.isnan(n))]
+        if len(non_nan_names) != len(set(non_nan_names)):
             raise ValueError("Duplicate names are not allowed.")
+
         if not (
             is_list_like(names, allow_sets=False) or isinstance(names, abc.KeysView)
         ):
```

This fix explicitly checks for duplicate NaN values before using the set-based check for other duplicates.