# Bug Report: pandas.io.formats.format.format_percentiles Unique Inputs Produce Duplicate Outputs

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `format_percentiles` function violates its documented guarantee that "if any two elements of percentiles differ, they remain different after rounding". When given unique percentile values that are extremely close to the same integer (e.g., `[0.0, 7.506590166045388e-253]`), the function produces duplicate outputs (`['0%', '0%']`).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.formats.format import format_percentiles

@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=20, unique=True))
@settings(max_examples=1000)
def test_format_percentiles_unique_inputs_remain_unique(percentiles):
    """
    Property from docstring: "if any two elements of percentiles differ,
    they remain different after rounding"
    """
    formatted = format_percentiles(percentiles)
    assert len(formatted) == len(set(formatted)), \
        f"Unique inputs produced duplicate outputs: {percentiles} -> {formatted}"
```

**Failing input**: `[0.0, 7.506590166045388e-253]`

## Reproducing the Bug

```python
from pandas.io.formats.format import format_percentiles

percentiles = [0.0, 7.506590166045388e-253]
formatted = format_percentiles(percentiles)

print(f"Input: {percentiles}")
print(f"Output: {formatted}")

assert len(formatted) == len(set(formatted))
```

Output:
```
Input: [0.0, 7.506590166045388e-253]
Output: ['0%', '0%']
AssertionError: Unique inputs should produce unique outputs
```

## Why This Is A Bug

The function's docstring explicitly states: "Rounding precision is chosen so that: (1) if any two elements of `percentiles` differ, they remain different after rounding". This property is violated when percentiles are extremely close to the same value but technically different.

The bug occurs because the function has an early return path when all percentiles are considered "integers" (line: `if np.all(int_idx)`). In this case, it doesn't check for uniqueness or use the precision calculation based on `unique_pcts`, which would preserve distinctiveness.

## Fix

The fix is to remove the early return and always use the uniqueness-aware precision calculation:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1585,14 +1585,8 @@ def format_percentiles(
     raise ValueError("percentiles should all be in the interval [0,1]")

 percentiles = 100 * percentiles
+unique_pcts = np.unique(percentiles)
 prec = get_precision(percentiles)
-percentiles_round_type = percentiles.round(prec).astype(int)
-
-int_idx = np.isclose(percentiles_round_type, percentiles)
-
-if np.all(int_idx):
-    out = percentiles_round_type.astype(str)
-    return [i + "%" for i in out]
-
-unique_pcts = np.unique(percentiles)
 prec = get_precision(unique_pcts)
+percentiles_round_type = percentiles.round(prec).astype(int)
+int_idx = np.isclose(percentiles_round_type, percentiles)
 out = np.empty_like(percentiles, dtype=object)
```