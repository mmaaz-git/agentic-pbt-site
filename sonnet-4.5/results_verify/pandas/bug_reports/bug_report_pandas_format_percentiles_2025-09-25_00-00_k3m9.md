# Bug Report: pandas.io.formats.format.format_percentiles - Numerical Issues with Small Percentiles

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `format_percentiles` function violates its documented contract and produces invalid output when given very small percentile values. The function's docstring promises that (1) different inputs remain different after formatting, and (2) no entry is rounded to 0% or 100% unless exactly 0 or 1. Both properties are violated for small percentile values due to numerical overflow in precision calculation.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.methods.describe import format_percentiles

@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=20))
def test_format_percentiles_uniqueness_property(percentiles):
    unique_percentiles = list(set(percentiles))
    assume(len(unique_percentiles) >= 2)

    result = format_percentiles(unique_percentiles)

    unique_results = set(result)
    assert len(unique_results) == len(unique_percentiles), \
        f"Uniqueness not preserved: {unique_percentiles} -> {result}"

@given(st.lists(st.floats(min_value=1e-10, max_value=1.0 - 1e-10, allow_nan=False, allow_infinity=False), min_size=1, max_size=20))
def test_format_percentiles_never_rounds_to_zero_or_hundred(percentiles):
    assume(all(0 < p < 1 for p in percentiles))

    result = format_percentiles(percentiles)

    for s in result:
        assert s != "0%", f"Rounded to 0% for input {percentiles}"
        assert s != "100%", f"Rounded to 100% for input {percentiles}"
```

**Failing inputs**:
- `[0.0, 5e-324]` produces `['nan%', 'nan%']`
- `[0.0, 1.401298464324817e-45]` produces `['0%', '0%']`
- `[1e-10]` produces `['0%']`

## Reproducing the Bug

```python
from pandas.core.methods.describe import format_percentiles

result1 = format_percentiles([0.0, 5e-324])
print(result1)

result2 = format_percentiles([0.0, 1.401298464324817e-45])
print(result2)

result3 = format_percentiles([1e-10])
print(result3)
```

## Why This Is A Bug

The function's docstring explicitly documents these properties:

> "Rounding precision is chosen so that: (1) if any two elements of `percentiles` differ, they remain different after rounding (2) no entry is *rounded* to 0% or 100%."

The examples above violate both properties:
1. Different inputs `[0.0, 1.401298464324817e-45]` become identical `['0%', '0%']`
2. Non-zero input `1e-10` is rounded to `'0%'`
3. Worst of all, valid inputs produce invalid output `'nan%'`

The root cause is in `get_precision()` which computes `prec = -np.floor(np.log10(np.min(diff))).astype(int)`. For very small differences, this can:
- Produce extremely large precision values (e.g., 322) causing numerical overflow when calling `array.round(prec)`
- Result in precision too high to distinguish values when converted to int/string

## Fix

The fix should add bounds checking to `get_precision()` to prevent numerical overflow and ensure the precision stays within reasonable limits:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1611,7 +1611,10 @@ def get_precision(array: np.ndarray | Sequence[float]) -> int:
     to_end = 100 - array[-1] if array[-1] < 100 else None
     diff = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
     diff = abs(diff)
-    prec = -np.floor(np.log10(np.min(diff))).astype(int)
+    min_diff = np.min(diff)
+    if min_diff == 0 or min_diff < 1e-15:
+        prec = 15
+    else:
+        prec = min(15, -np.floor(np.log10(min_diff)).astype(int))
     prec = max(1, prec)
     return prec
```

This caps precision at 15 decimal places (sufficient for float64) and handles edge cases where differences are too small to distinguish reliably, preventing overflow while maintaining reasonable formatting behavior.