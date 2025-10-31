# Bug Report: pandas.core.methods.describe format_percentiles creates duplicate labels

**Target**: `pandas.core.methods.describe.format_percentiles`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`format_percentiles` can produce duplicate labels when given percentiles that are distinct floats but round to the same formatted string. This causes `df.describe()` to crash with "cannot reindex on an axis with duplicate labels".

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume
from pandas.core.methods.describe import format_percentiles, describe_numeric_1d


@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=100),
    st.lists(st.floats(min_value=0.01, max_value=0.99), min_size=1, max_size=5).map(lambda x: sorted(list(set(x))))
)
@settings(max_examples=500)
def test_describe_numeric_min_max_bounds(data, percentiles):
    series = pd.Series(data)
    assume(series.count() > 0)
    assume(len(percentiles) > 0)

    result = describe_numeric_1d(series, percentiles)

    formatted_pcts = format_percentiles(percentiles)
    for pct_label in formatted_pcts:
        pct_val = float(result[pct_label])
```

**Failing input**: `percentiles=[0.01, 0.010000000000000002]`

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

percentiles = [0.01, 0.010000000000000002]

result = df.describe(percentiles=percentiles)
```

**Output:**
```
ValueError: cannot reindex on an axis with duplicate labels
```

**Root cause:**
```python
from pandas.core.methods.describe import format_percentiles

percentiles = [0.01, 0.010000000000000002]
formatted = format_percentiles(percentiles)

print(formatted)
```

**Output:**
```
['1%', '1%']
```

## Why This Is A Bug

The docstring for `format_percentiles` explicitly states:

> "Rounding precision is chosen so that: (1) if any two elements of percentiles differ, they remain different after rounding"

However, when percentiles differ by less than the rounding precision (e.g., `0.01` and `0.01 + 2e-18`), they are:
1. Distinct floats (pass `set()` and `np.unique()` checks)
2. Pass validation in `_refine_percentiles`
3. But get formatted to the same label ('1%')

This creates duplicate index labels in the describe output, causing pandas operations to fail.

## Fix

The bug occurs because `get_precision` doesn't account for the case where distinct percentiles are closer together than the achievable decimal precision. The function should either:

1. **Option A**: Check for duplicate labels after formatting and raise a clear error
2. **Option B**: When precision calculation results in duplicates, increase precision iteratively until all labels are distinct

Here's a patch for Option A (simpler and clearer):

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1610,7 +1610,13 @@ def format_percentiles(

     out[~int_idx] = percentiles[~int_idx].round(prec).astype(str)
-    return [i + "%" for i in out]
+    result = [i + "%" for i in out]
+
+    if len(result) != len(set(result)):
+        raise ValueError(
+            "percentiles are too close together to be formatted distinctly. "
+            "Please ensure percentiles differ by at least 0.01 (1%)"
+        )
+    return result
```

This fix ensures that the documented property is maintained: distinct percentiles either remain distinct after formatting, or the function raises a clear error explaining why it can't format them distinctly.