# Bug Report: pandas.tseries.frequencies Reflexivity Inconsistency

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_subperiod` and `is_superperiod` functions are inconsistent in their handling of reflexivity (whether a frequency is a sub/superperiod of itself). Some frequencies return `True` for both operations when compared to themselves, while others return `False` for both, and annual frequencies return `True` for `is_superperiod` but `False` for `is_subperiod`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.tseries.frequencies import is_subperiod, is_superperiod

freq_strings = st.sampled_from([
    'D', 'h', 'min', 's', 'ms', 'us', 'ns', 'B', 'C',
    'W', 'W-MON', 'M', 'BM', 'Q', 'Q-JAN',
    'Y', 'Y-JAN',
])

@given(freq_strings)
def test_reflexivity_consistency(freq):
    is_super = is_superperiod(freq, freq)
    is_sub = is_subperiod(freq, freq)
    assert is_super == is_sub, f"Inconsistent reflexivity for {freq}"
```

**Failing inputs**: `'Y'`, `'Y-JAN'`, `'Y-FEB'`, `'Y-MAR'`, etc.

## Reproducing the Bug

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod

test_freqs = [
    'D', 'h', 'min', 's', 'B', 'C', 'W',
    'M', 'BM', 'Q', 'Q-JAN', 'Y', 'Y-JAN',
]

for freq in test_freqs:
    is_super = is_superperiod(freq, freq)
    is_sub = is_subperiod(freq, freq)
    print(f"{freq:10s}: super={str(is_super):5s}, sub={str(is_sub):5s}")
```

Output:
```
D         : super=True , sub=True
h         : super=True , sub=True
min       : super=True , sub=True
s         : super=True , sub=True
B         : super=True , sub=True
C         : super=True , sub=True
W         : super=True , sub=True
M         : super=False, sub=False
BM        : super=False, sub=False
Q         : super=False, sub=False
Q-JAN     : super=False, sub=False
Y         : super=True , sub=False  ← Inconsistent!
Y-JAN     : super=True , sub=False  ← Inconsistent!
```

## Why This Is A Bug

1. **Symmetry violation**: As documented in a separate bug report, the inconsistent handling of annual frequencies violates the fundamental symmetry property that `is_superperiod(a, b) == is_subperiod(b, a)`.

2. **Inconsistent behavior**: The functions handle reflexivity differently for different frequency types:
   - Day/hour/minute/second frequencies: Both return `True`
   - Monthly/quarterly frequencies: Both return `False`
   - Annual frequencies: `is_superperiod` returns `True`, `is_subperiod` returns `False`

3. **Unclear semantics**: Without consistent reflexivity behavior, it's unclear whether a frequency should be considered "compatible" with itself for conversion purposes.

## Fix

The root cause is that the functions don't explicitly handle the reflexive case for all frequency types. The fix should ensure consistent behavior across all frequencies. Looking at similar functions in the codebase, weekly frequencies explicitly include `source` in their target set (line 524 in `is_superperiod`), suggesting that reflexivity is intended.

For `is_subperiod`:

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -453,6 +453,10 @@ def is_subperiod(source, target) -> bool:
     target = _maybe_coerce_freq(target)

     if _is_annual(target):
+        if _is_annual(source):
+            return get_rule_month(source) == get_rule_month(target)
+
         if _is_quarterly(source):
             return _quarter_months_conform(
                 get_rule_month(source), get_rule_month(target)
@@ -460,8 +464,12 @@ def is_subperiod(source, target) -> bool:
             )
         return source in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
     elif _is_quarterly(target):
+        if _is_quarterly(source):
+            return source == target
         return source in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
     elif _is_monthly(target):
+        if _is_monthly(source):
+            return source == target
         return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
```

For `is_superperiod`:

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -518,8 +522,12 @@ def is_superperiod(source, target) -> bool:
             return _quarter_months_conform(smonth, tmonth)
         return target in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
     elif _is_quarterly(source):
+        if _is_quarterly(target):
+            return source == target
         return target in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
     elif _is_monthly(source):
+        if _is_monthly(target):
+            return source == target
         return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
```

This ensures that all frequency types have consistent reflexive behavior, matching the pattern already established for weekly frequencies.