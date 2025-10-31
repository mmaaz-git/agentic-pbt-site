# Bug Report: pandas.io.formats.format_percentiles Violates Documented Contract

**Target**: `pandas.io.formats.format.format_percentiles`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_percentiles` function violates its documented contract by rounding values to 0% or 100% when the docstring explicitly states it should not (except when the input is exactly 0 or 1).

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from pandas.io.formats.format import format_percentiles

@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False,
                          allow_infinity=False), min_size=1, max_size=20))
@settings(max_examples=500)
def test_format_percentiles_no_rounding_to_zero_or_hundred(percentiles):
    assume(len(percentiles) > 0)
    assume(all(0 <= p <= 1 for p in percentiles))

    result = format_percentiles(percentiles)

    for i, (p, formatted) in enumerate(zip(percentiles, result)):
        if p != 0.0:
            assert formatted != "0%", f"Non-zero percentile {p} was rounded to 0%"
        if p != 1.0:
            assert formatted != "100%", f"Non-one percentile {p} was rounded to 100%"
```

**Failing input**: `[0.99999, 0.0]`

## Reproducing the Bug

```python
from pandas.io.formats.format import format_percentiles

result = format_percentiles([0.99999, 0.0])
print(result)

assert result[0] != "100%", f"Expected 99.999 to not round to 100%, got {result[0]}"
```

Output:
```
['100%', '0%']
AssertionError: Expected 99.999 to not round to 100%, got 100%
```

## Why This Is A Bug

The function's docstring explicitly states:

> **Notes**
> Rounding precision is chosen so that: (1) if any two elements of
> `percentiles` differ, they remain different after rounding
> (2) **no entry is *rounded* to 0% or 100%.**

The docstring also provides an example showing this property:

```python
>>> format_percentiles([0.01999, 0.02001, 0.5, 0.666666, 0.9999])
['1.999%', '2.001%', '50%', '66.667%', '99.99%']
```

Note that 0.9999 (99.99%) is NOT rounded to 100%.

However, `format_percentiles([0.99999, 0.0])` returns `['100%', '0%']`, violating this contract.

The bug occurs because of line 1603 in `/pandas/io/formats/format.py`:

```python
out[int_idx] = percentiles[int_idx].round().astype(int).astype(str)
```

This line unconditionally rounds values to the nearest integer when they are "close" to integers (as determined by `np.isclose`), without checking whether this would produce 0 or 100.

## Fix

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1600,7 +1600,12 @@ def format_percentiles(
     unique_pcts = np.unique(percentiles)
     prec = get_precision(unique_pcts)
     out = np.empty_like(percentiles, dtype=object)
-    out[int_idx] = percentiles[int_idx].round().astype(int).astype(str)
+
+    # Round to integers, but avoid rounding to 0 or 100
+    rounded_ints = percentiles[int_idx].round().astype(int)
+    # Only use integer format if it doesn't produce 0 or 100 (unless input is exactly 0 or 100)
+    safe_int_idx = (rounded_ints != 0) | (percentiles[int_idx] == 0) & (rounded_ints != 100) | (percentiles[int_idx] == 100)
+    out[int_idx] = np.where(safe_int_idx, rounded_ints.astype(str), percentiles[int_idx].round(prec).astype(str))

     out[~int_idx] = percentiles[~int_idx].round(prec).astype(str)
     return [i + "%" for i in out]
```

Actually, a simpler fix that better respects the function's intent:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1600,7 +1600,14 @@ def format_percentiles(
     unique_pcts = np.unique(percentiles)
     prec = get_precision(unique_pcts)
     out = np.empty_like(percentiles, dtype=object)
-    out[int_idx] = percentiles[int_idx].round().astype(int).astype(str)
+
+    # Round to integers, but avoid rounding to 0 or 100 unless input is exactly 0 or 100
+    for i in np.where(int_idx)[0]:
+        rounded = int(percentiles[i].round())
+        if (rounded == 0 and percentiles[i] != 0) or (rounded == 100 and percentiles[i] != 100):
+            out[i] = f"{percentiles[i].round(prec)}"
+        else:
+            out[i] = str(rounded)

     out[~int_idx] = percentiles[~int_idx].round(prec).astype(str)
     return [i + "%" for i in out]
```