# Bug Report: pandas.io.formats.format.format_percentiles Non-Integer Without Decimal Place

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_percentiles` function violates its documented guarantee that "Any non-integer is always rounded to at least 1 decimal place". When given certain non-integer percentiles (e.g., `0.8899967487632947` which is `88.99967487632946%`), the function produces output without a decimal place (`'89%'`).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.formats.format import format_percentiles

@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=1000)
def test_format_percentiles_non_integer_has_decimal(percentile):
    """
    Property from docstring: "Any non-integer is always rounded to at least 1 decimal place"
    """
    formatted = format_percentiles([percentile])
    result = formatted[0]

    percent_value = percentile * 100
    is_integer_percent = abs(percent_value - round(percent_value)) < 1e-10

    if not is_integer_percent:
        assert '.' in result or ',' in result, \
            f"Non-integer percentile {percentile} formatted without decimal: {result}"
```

**Failing input**: `0.8899967487632947`

## Reproducing the Bug

```python
from pandas.io.formats.format import format_percentiles

percentile = 0.8899967487632947
formatted = format_percentiles([percentile])

print(f"Input: {percentile}")
print(f"Percent value: {percentile * 100}%")
print(f"Output: {formatted[0]}")

is_integer = abs((percentile * 100) - round(percentile * 100)) < 1e-10
print(f"Is integer percent: {is_integer}")

assert '.' in formatted[0]
```

Output:
```
Input: 0.8899967487632947
Percent value: 88.99967487632946%
Output: 89%
Is integer percent: False
AssertionError: Non-integer should have decimal place
```

## Why This Is A Bug

The function's docstring explicitly states: "Any non-integer is always rounded to at least 1 decimal place". However, the function uses `np.isclose()` to determine if a percentile should be treated as an integer, which can incorrectly classify non-integers that are close to integers after rounding.

The issue is in this logic:
```python
percentiles_round_type = percentiles.round(prec).astype(int)
int_idx = np.isclose(percentiles_round_type, percentiles)
```

When a percentile like `88.99967487632946%` is rounded at a certain precision, it becomes close enough to `89` that `np.isclose()` returns True, causing it to be formatted as an integer without a decimal place.

## Fix

The fix requires using a stricter check to determine if a percentile is truly an integer value, rather than just "close" to an integer after rounding:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1589,7 +1589,10 @@ def format_percentiles(
 prec = get_precision(percentiles)
 percentiles_round_type = percentiles.round(prec).astype(int)

-int_idx = np.isclose(percentiles_round_type, percentiles)
+# Check if percentile is truly an integer (e.g., 0.5 -> 50%), not just close after rounding
+int_idx = np.abs(percentiles - np.round(percentiles)) < 1e-10

 if np.all(int_idx):
     out = percentiles_round_type.astype(str)
```