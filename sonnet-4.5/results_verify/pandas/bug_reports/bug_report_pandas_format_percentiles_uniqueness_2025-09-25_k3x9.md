# Bug Report: pandas.core.methods.describe.format_percentiles Uniqueness Preservation Violation

**Target**: `pandas.core.methods.describe.format_percentiles`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `format_percentiles` function violates its documented contract by failing to preserve uniqueness when formatting very small percentile values (around 1e-38 or smaller). The function's docstring explicitly promises "if any two elements of percentiles differ, they remain different after rounding", but this property fails for extremely small but distinct values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@settings(max_examples=500)
@given(st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=10
).filter(lambda lst: len(set(lst)) == len(lst)))
def test_format_percentiles_uniqueness_preservation(percentiles):
    from pandas.core.methods.describe import format_percentiles
    result = format_percentiles(percentiles)
    assert len(set(result)) == len(result), \
        f"Uniqueness not preserved: {percentiles} -> {result}"
```

**Failing input**: `[0.0, 1.175494351e-38]`

## Reproducing the Bug

```python
from pandas.core.methods.describe import format_percentiles

percentiles = [0.0, 1.175494351e-38]
result = format_percentiles(percentiles)

print(f"Input: {percentiles}")
print(f"Output: {result}")
print(f"Expected: Two different formatted strings")
print(f"Actual: {result} (both are '0%')")
```

Output:
```
Input: [0.0, 1.175494351e-38]
Output: ['0%', '0%']
Expected: Two different formatted strings
Actual: ['0%', '0%'] (both are '0%')
```

## Why This Is A Bug

The function's docstring explicitly states:

> "Rounding precision is chosen so that: (1) if any two elements of `percentiles` differ, they remain different after rounding"

This is an unconditional promise with no caveats about value ranges. The two input values `0.0` and `1.175494351e-38` are distinct, so the formatted output should also be distinct.

The root cause is in the `get_precision` helper function in `pandas.io.formats.format`, which computes the required decimal precision as:

```python
prec = -np.floor(np.log10(np.min(diff))).astype(int)
```

For very small differences (e.g., 1e-38), this produces precision values like 38, which exceed the numerical precision limits of IEEE 754 double precision floats (~15-17 significant digits). When the code attempts to round with such high precision, numerical overflow occurs, causing the rounding operation to produce identical results for distinct inputs.

## Fix

Add a maximum precision cap to prevent numerical overflow. Since IEEE 754 double precision provides about 15-17 significant decimal digits, capping precision at 15 is reasonable:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1612,7 +1612,7 @@ def get_precision(array: np.ndarray | Sequence[float]) -> int:
     diff = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
     diff = abs(diff)
     prec = -np.floor(np.log10(np.min(diff))).astype(int)
-    prec = max(1, prec)
+    prec = max(1, min(prec, 15))
     return prec
```

This ensures that the precision stays within the limits of floating-point arithmetic, preventing overflow while still handling realistic percentile differences correctly.