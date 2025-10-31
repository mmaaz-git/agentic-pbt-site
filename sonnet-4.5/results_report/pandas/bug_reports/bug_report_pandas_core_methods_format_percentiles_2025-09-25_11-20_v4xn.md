# Bug Report: pandas.core.methods.describe.format_percentiles Produces Invalid Output

**Target**: `pandas.core.methods.describe.format_percentiles`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `format_percentiles` function produces invalid output strings (`"inf%"`, `"nan%"`) and violates its documented uniqueness preservation property when given very small but valid percentile values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from pandas.core.methods.describe import format_percentiles
import numpy as np

@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=20))
def test_format_percentiles_uniqueness_preservation(percentiles):
    unique_input = np.unique(percentiles)
    assume(len(unique_input) >= 2)

    result = format_percentiles(percentiles)
    unique_output = set(result)

    assert len(unique_input) == len(unique_output), \
        f"Uniqueness not preserved: {len(unique_input)} unique inputs -> {len(unique_output)} unique outputs"
```

**Failing input**: `[0.0, 2.225073858507203e-309]`

## Reproducing the Bug

```python
from pandas.core.methods.describe import format_percentiles

percentiles = [0.0, 2.225073858507203e-309]
result = format_percentiles(percentiles)
print(result)

percentiles2 = [0.9374708737308688, 0.5, 0.22976323147538885, 0.8834939856757813, 2.225073858507203e-309]
result2 = format_percentiles(percentiles2)
print(result2)

percentiles3 = [5e-324, 9.889517165452854e-55, 0.027853332816048855, 0.5]
result3 = format_percentiles(percentiles3)
print(result3)
```

Output:
```
['0%', '0%']
['inf%', '50%', 'inf%', 'inf%', '0%']
['nan%', 'nan%', 'nan%', 'nan%']
```

## Why This Is A Bug

The function's docstring explicitly states:

> "Rounding precision is chosen so that: (1) if any two elements of percentiles differ, they remain different after rounding"

However, `[0.0, 2.225073858507203e-309]` are two different values that both get formatted as `'0%'`, violating this contract.

More critically, the function produces completely invalid output strings like `"inf%"` and `"nan%"` which are not valid percentage representations. These inputs are valid percentiles (all in the range [0, 1]), and the function should handle them correctly.

The root cause is that `get_precision` computes precision as `-floor(log10(min_diff))`. For very small differences (like `2.225e-309`), this yields extremely large precision values (e.g., 307), which cause numerical overflow when passed to `round()` and `astype(int)`.

## Fix

Add a maximum precision cap to prevent numerical overflow. Float64 precision is ~15-17 decimal digits, so capping at 15 is reasonable:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1612,7 +1612,8 @@ def get_precision(array: np.ndarray | Sequence[float]) -> int:
     diff = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
     diff = abs(diff)
     prec = -np.floor(np.log10(np.min(diff))).astype(int)
-    prec = max(1, prec)
+    # Cap precision to avoid overflow with very small differences
+    prec = max(1, min(prec, 15))
     return prec
```

This ensures that:
1. No numerical overflow occurs
2. Output is always valid percentage strings
3. The function gracefully handles edge cases with very small percentiles

For the uniqueness preservation guarantee to hold for extremely small values, the docstring should be updated to acknowledge the precision limitation, or the implementation should use a different approach (e.g., scientific notation for very small percentiles).