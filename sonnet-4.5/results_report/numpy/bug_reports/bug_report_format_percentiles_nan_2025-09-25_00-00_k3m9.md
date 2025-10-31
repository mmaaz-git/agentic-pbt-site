# Bug Report: format_percentiles produces 'nan%' output

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`format_percentiles()` produces invalid 'nan%' output when given valid percentiles that include extremely small values mixed with normal values. This occurs due to numeric overflow in the precision calculation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.io.formats.format as fmt

@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1))
@settings(max_examples=1000)
def test_format_percentiles_all_end_with_percent(percentiles):
    """
    Property: all formatted strings should end with '%'
    and not contain 'nan' or 'inf'
    """
    formatted = fmt.format_percentiles(percentiles)
    for f in formatted:
        assert f.endswith('%'), f"Formatted value '{f}' does not end with '%'"
        assert 'nan' not in f.lower(), f"Formatted value '{f}' contains 'nan'"
```

**Failing input**: `[0.625, 5e-324]`

## Reproducing the Bug

```python
import pandas.io.formats.format as fmt

result = fmt.format_percentiles([0.625, 5e-324])
print(result)
```

Output:
```
['nan%', '0%']
```

## Why This Is A Bug

The function `format_percentiles` is documented to return formatted percentile strings. The output 'nan%' is not a valid percentage string and would cause errors in downstream code that expects numeric values or valid formatting.

The bug occurs in the `get_precision` helper function. When the input contains extremely small values, the difference calculation produces very small or zero values. Taking `log10` of these produces `-inf` or very large negative numbers, which when negated and cast to int causes numeric overflow:

```python
# From get_precision function (line 1614)
prec = -np.floor(np.log10(np.min(diff))).astype(int)
```

When `np.min(diff)` is extremely small (like 4.94e-322), `log10` returns a very large negative number. This causes:
1. The precision value to overflow when cast to int
2. Subsequent rounding operations to produce NaN values
3. The final output to contain 'nan%'

## Fix

The `get_precision` function should handle the case where differences are extremely small or zero by capping the precision at a reasonable maximum value.

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1611,7 +1611,13 @@ def get_precision(array: np.ndarray | Sequence[float]) -> int:
     to_end = 100 - array[-1] if array[-1] < 100 else None
     diff = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
     diff = abs(diff)
-    prec = -np.floor(np.log10(np.min(diff))).astype(int)
+    min_diff = np.min(diff)
+    if min_diff == 0 or not np.isfinite(min_diff):
+        prec = 15  # Maximum reasonable precision for float display
+    else:
+        log_val = np.log10(min_diff)
+        if not np.isfinite(log_val) or log_val < -15:
+            prec = 15
+        else:
+            prec = -np.floor(log_val).astype(int)
     prec = max(1, prec)
     return prec
```