# Bug Report: pandas.io.formats.format.format_percentiles Rounds Non-Zero to 0%

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_percentiles` function rounds very small non-zero percentiles to `0%`, violating its documented contract that "no entry is *rounded* to 0% or 100%" unless already equal to it. Additionally, it violates the property that "any non-integer is always rounded to at least 1 decimal place."

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from pandas.io.formats.format import format_percentiles


@given(
    percentile=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
def test_format_percentiles_non_integer_has_decimal(percentile):
    assume(percentile not in [0.0, 1.0])
    assume(not (percentile * 100).is_integer())

    result = format_percentiles([percentile])

    assert len(result) == 1
    value_str = result[0].rstrip('%')
    assert '.' in value_str, f"Non-integer {percentile} should have decimal: {result[0]}"
```

**Failing input**: `1.401298464324817e-45`

## Reproducing the Bug

```python
from pandas.io.formats.format import format_percentiles

percentile = 1.401298464324817e-45
result = format_percentiles([percentile])

print(f"Input: {percentile}")
print(f"Output: {result[0]}")
print(f"Is zero: {percentile == 0.0}")
print(f"Is integer: {(percentile * 100).is_integer()}")

assert result[0] == '0%'
```

Output:
```
Input: 1.401298464324817e-45
Output: 0%
Is zero: False
Is integer: False
```

## Why This Is A Bug

The function's docstring explicitly states:

1. "no entry is *rounded* to 0% or 100%." (unless already equal to it)
2. "Any non-integer is always rounded to at least 1 decimal place."

However, the input `1.401298464324817e-45` is:
- Not equal to 0.0
- Not an integer when multiplied by 100

Yet it's formatted as `0%` with no decimal place, violating both documented properties.

While this is an extreme edge case with very small numbers, the function should either:
1. Use enough precision to display a non-zero value (e.g., `0.0%` or with scientific notation)
2. Or explicitly document that extremely small values may be rounded to `0%`

## Fix

The issue occurs because `get_precision` doesn't handle extremely small differences well, leading to precision overflow. The function should add a check for values very close to 0 or 1:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1587,6 +1587,15 @@ def format_percentiles(
         raise ValueError("percentiles should all be in the interval [0,1]")

     percentiles = 100 * percentiles
+
+    # Ensure non-zero values near 0 or 100 have at least one decimal place
+    # to avoid rounding exactly to 0% or 100%
+    MIN_THRESHOLD = 1e-10  # Approximately the smallest value that rounds to non-zero
+    for i, p in enumerate(percentiles):
+        if 0 < p < MIN_THRESHOLD or 100 - MIN_THRESHOLD < p < 100:
+            # Force at least 1 decimal place for values very close to 0/100
+            out = np.array([f"{p:.1f}%" for p in percentiles], dtype=object)
+            return list(out)
+
     prec = get_precision(percentiles)
     percentiles_round_type = percentiles.round(prec).astype(int)