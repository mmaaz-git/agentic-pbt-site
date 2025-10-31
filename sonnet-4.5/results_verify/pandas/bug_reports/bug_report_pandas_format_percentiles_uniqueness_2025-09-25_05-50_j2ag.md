# Bug Report: pandas.io.formats.format_percentiles Uniqueness Violation

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`format_percentiles` violates its documented guarantee to preserve uniqueness when formatting percentiles. Two different percentile values can produce identical formatted strings, contradicting the function's explicit promise that "if any two elements of percentiles differ, they remain different after rounding."

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from pandas.io.formats.format import format_percentiles

@settings(max_examples=1000)
@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=100))
def test_format_percentiles_uniqueness_preservation(percentiles):
    unique_input = np.unique(percentiles)
    assume(len(unique_input) > 1)

    formatted = format_percentiles(percentiles)
    unique_input_formatted = format_percentiles(unique_input)
    unique_output = list(dict.fromkeys(unique_input_formatted))

    assert len(unique_output) == len(unique_input), \
        f"Unique percentiles should remain unique after formatting. " \
        f"Input had {len(unique_input)} unique values but output has {len(unique_output)}."
```

**Failing input**: `[0.01, 0.0100001]`

## Reproducing the Bug

```python
from pandas.io.formats.format import format_percentiles

percentiles = [0.01, 0.0100001]
result = format_percentiles(percentiles)

print(f"Input:  {percentiles}")
print(f"Output: {result}")
print(f"Unique inputs:  {len(set(percentiles))}")
print(f"Unique outputs: {len(set(result))}")
```

Output:
```
Input:  [0.01, 0.0100001]
Output: ['1%', '1%']
Unique inputs:  2
Unique outputs: 1
```

## Why This Is A Bug

The function's docstring explicitly states:

> Rounding precision is chosen so that: (1) if any two elements of
> ``percentiles`` differ, they remain different after rounding

However, when percentiles differ by less than approximately 1e-7 (as fractions) or 1e-5 (as percentages), the function fails to preserve this uniqueness. The root cause is on line 1592 of `/pandas/io/formats/format.py`:

```python
int_idx = np.isclose(percentiles_round_type, percentiles)
```

The `np.isclose` function uses default tolerances (rtol=1e-05, atol=1e-08), which are too lenient for the precision-preserving guarantee. For percentiles like `[1.0, 1.00001]` (after multiplication by 100), both values are considered "close" to integer 1, so both get formatted as `'1%'`, losing the distinction.

## Fix

The issue is that `np.isclose` with default tolerances is too permissive. The function should use exact equality or much tighter tolerances. Here's a patch:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1589,7 +1589,7 @@ def format_percentiles(
     percentiles = 100 * percentiles
     prec = get_precision(percentiles)
     percentiles_round_type = percentiles.round(prec).astype(int)
-    int_idx = np.isclose(percentiles_round_type, percentiles)
+    int_idx = (percentiles_round_type == percentiles)

     if np.all(int_idx):
         out = percentiles_round_type.astype(str)
```

Alternatively, if floating-point tolerance is necessary, use much tighter tolerances that won't violate the uniqueness guarantee:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1589,7 +1589,7 @@ def format_percentiles(
     percentiles = 100 * percentiles
     prec = get_precision(percentiles)
     percentiles_round_type = percentiles.round(prec).astype(int)
-    int_idx = np.isclose(percentiles_round_type, percentiles)
+    int_idx = np.isclose(percentiles_round_type, percentiles, rtol=0, atol=1e-10)

     if np.all(int_idx):
         out = percentiles_round_type.astype(str)
```