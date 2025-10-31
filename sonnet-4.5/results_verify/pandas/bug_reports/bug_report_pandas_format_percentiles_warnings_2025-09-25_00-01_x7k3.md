# Bug Report: pandas.io.formats.format.format_percentiles RuntimeWarnings with Duplicate Values

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`format_percentiles` produces RuntimeWarnings (divide by zero, invalid cast, overflow) when all input percentiles are identical, despite producing correct output. This is inefficient and pollutes the warning stream.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.io.formats.format as fmt
import warnings
import pytest


@given(
    value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    count=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=500)
def test_format_percentiles_no_warnings_for_duplicates(value, count):
    """When all percentiles are the same, the function should not produce warnings."""
    percentiles = [value] * count

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fmt.format_percentiles(percentiles)

        runtime_warnings = [warning for warning in w
                          if issubclass(warning.category, RuntimeWarning)]

        assert len(runtime_warnings) == 0, \
            f"Should not produce warnings for duplicate values: {[str(w.message) for w in runtime_warnings]}"
```

**Failing input**: `value=0.0, count=2` (or any duplicate values)

## Reproducing the Bug

```python
import pandas.io.formats.format as fmt
import warnings

warnings.simplefilter("always")
result = fmt.format_percentiles([0.5, 0.5, 0.5])
```

**Warnings produced:**
```
RuntimeWarning: divide by zero encountered in log10
RuntimeWarning: invalid value encountered in cast
RuntimeWarning: overflow encountered in scalar negative
```

**Root cause**: The `get_precision` helper function computes differences between consecutive percentiles using `np.ediff1d`. When all values are identical, all differences are zero. It then calls `np.log10(0)`, which produces `-inf`, leading to a cascade of warnings in subsequent operations.

Relevant code from `get_precision`:
```python
diff = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
diff = abs(diff)
prec = -np.floor(np.log10(np.min(diff))).astype(int)  # log10(0) = -inf
```

## Why This Is A Bug

1. **Performance**: The function performs unnecessary expensive computations (log10, floor, etc.) that generate warnings for a trivial case
2. **User experience**: Users see cryptic warnings even though the output is correct
3. **Best practice**: Functions should handle expected inputs (including duplicates) without warnings

While the function produces correct output, generating warnings for valid input is a code smell indicating poor handling of edge cases.

## Fix

Add a check for duplicate values before calling `get_precision`:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1578,6 +1578,10 @@ def format_percentiles(
     if not is_numeric_dtype(percentiles) or not np.all(percentiles >= 0) or not np.all(percentiles <= 1):
         raise ValueError("percentiles should all be in the interval [0,1]")

+    # Handle the case where all percentiles are identical
+    if len(np.unique(percentiles)) == 1:
+        return [f"{int(percentiles[0] * 100)}%" for _ in percentiles]
+
     percentiles = 100 * percentiles
     prec = get_precision(percentiles)
     percentiles_round_type = percentiles.round(prec).astype(int)
```

Alternatively, fix `get_precision` to handle zero differences:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1610,6 +1610,10 @@ def get_precision(array: np.ndarray | Sequence[float]) -> int:
     to_end = 100 - array[-1] if array[-1] < 100 else None
     diff = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
     diff = abs(diff)
+    min_diff = np.min(diff)
+    if min_diff == 0:
+        # All values are identical, no precision needed
+        return 0
     prec = -np.floor(np.log10(np.min(diff))).astype(int)
     prec = max(1, prec)
     return prec
```