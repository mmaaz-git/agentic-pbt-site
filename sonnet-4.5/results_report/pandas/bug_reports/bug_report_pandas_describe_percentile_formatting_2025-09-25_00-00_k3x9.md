# Bug Report: pandas.core.methods.describe - Percentile Formatting Overflow

**Target**: `pandas.core.methods.describe` (format_percentiles function in pandas.io.formats.format)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `describe()` is called with extremely small percentile values (e.g., 5e-324), the percentile formatting logic fails, producing invalid "nan%" labels and violating the documented invariant that the median (50%) should always be included.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd

@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=5, max_size=100),
    percentiles=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=10)
)
@settings(max_examples=500)
def test_describe_median_always_included(values, percentiles):
    percentiles = [p for p in percentiles if 0 <= p <= 1]
    assume(len(percentiles) > 0)
    assume(len(set(percentiles)) == len(percentiles))
    assume(0.5 not in percentiles)

    series = pd.Series(values)
    result = series.describe(percentiles=percentiles)

    assert '50%' in result.index, "Median (50%) should always be included in describe output"
```

**Failing input**: `percentiles=[5e-324]`

## Reproducing the Bug

```python
import pandas as pd

series = pd.Series([1, 2, 3, 4, 5])

result = series.describe(percentiles=[5e-324])
print(result.index.tolist())
```

**Output:**
```
['count', 'mean', 'std', 'min', 'nan%', 'nan%', 'max']
```

**Expected:**
```
['count', 'mean', 'std', 'min', '<proper_label>', '50%', 'max']
```

## Why This Is A Bug

1. **Violates documented invariant**: The code at `describe.py:404-406` explicitly states "median should always be included", but the median (50%) is missing from the output.

2. **Invalid percentile labels**: The formatting produces "nan%" labels instead of proper percentage strings.

3. **Accepts invalid results**: The function accepts percentiles in [0, 1] per its contract (line 1588 in format.py validates this), but produces malformed output for valid extreme values.

4. **Numerical overflow in precision calculation**: In `format.py:1614`, the precision calculation `prec = -np.floor(np.log10(np.min(diff))).astype(int)` produces extremely large values (e.g., 322) for tiny percentiles, causing overflow in subsequent rounding and type conversion operations.

## Fix

The root cause is in `pandas/io/formats/format.py` in the `get_precision` function. When percentiles are extremely small, the log10 calculation produces unbounded precision values. The fix should add bounds checking:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1611,7 +1611,8 @@ def get_precision(array: np.ndarray | Sequence[float]) -> int:
     to_end = 100 - array[-1] if array[-1] < 100 else None
     diff = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
     diff = abs(diff)
-    prec = -np.floor(np.log10(np.min(diff))).astype(int)
+    min_diff = max(np.min(diff), 1e-15)
+    prec = -np.floor(np.log10(min_diff)).astype(int)
     prec = max(1, prec)
     return prec
```

This bounds the minimum diff value to prevent extreme precision calculations, limiting precision to approximately 15 decimal places (reasonable for float64 precision).