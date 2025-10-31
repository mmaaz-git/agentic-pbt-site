# Bug Report: pandas.tseries.frequencies Reflexivity Violation

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod(X, X)` and `is_superperiod(X, X)` should always return True (a frequency is both a subperiod and superperiod of itself), but they return False for monthly ('M'), quarterly ('Q'), and some annual ('Y') frequencies.

## Property-Based Test

```python
import pandas.tseries.frequencies as freq
from hypothesis import given, strategies as st, settings

VALID_FREQS = ["D", "h", "B", "C", "M", "W", "Q", "Y", "min", "s", "ms", "us", "ns"]

@given(st.sampled_from(VALID_FREQS))
@settings(max_examples=50)
def test_subperiod_reflexive(freq_str):
    """
    Property: Reflexivity - is_subperiod(X, X) should always be True
    (a frequency is a subperiod of itself).
    """
    result = freq.is_subperiod(freq_str, freq_str)
    assert result, f"Reflexivity violated: is_subperiod({freq_str}, {freq_str}) returned False"

@given(st.sampled_from(VALID_FREQS))
@settings(max_examples=50)
def test_superperiod_reflexive(freq_str):
    """
    Property: Reflexivity - is_superperiod(X, X) should always be True
    (a frequency is a superperiod of itself).
    """
    result = freq.is_superperiod(freq_str, freq_str)
    assert result, f"Reflexivity violated: is_superperiod({freq_str}, {freq_str}) returned False"
```

**Failing inputs**: `freq_str='M'` for both tests

## Reproducing the Bug

```python
import pandas.tseries.frequencies as freq

test_freqs = ["M", "Q", "Y"]

for f in test_freqs:
    sub_result = freq.is_subperiod(f, f)
    super_result = freq.is_superperiod(f, f)
    print(f"is_subperiod('{f}', '{f}') = {sub_result}")
    print(f"is_superperiod('{f}', '{f}') = {super_result}")
    print()
```

Output:
```
is_subperiod('M', 'M') = False
is_superperiod('M', 'M') = False

is_subperiod('Q', 'Q') = False
is_superperiod('Q', 'Q') = False

is_subperiod('Y', 'Y') = False
is_superperiod('Y', 'Y') = True
```

## Why This Is A Bug

Reflexivity is a fundamental mathematical property: any frequency should be considered both a subperiod and superperiod of itself, since you can convert from a frequency to itself without any resampling.

The bug occurs because these functions don't explicitly handle the reflexive case for monthly, quarterly, and some annual frequencies. When `source == target` for these frequencies, the code falls through to `return False` at the end of the function instead of returning True.

Looking at the code:
- For 'M' (monthly): `_is_monthly(target)` is True, but 'M' is not in the returned set `{"D", "C", "B", "h", ...}`, so it returns False
- For 'Q' (quarterly): Similar issue - 'Q' is not in the returned set
- For 'Y' (annual): In `is_subperiod`, it checks `_is_quarterly(source)` first, which is False, then falls through to `return False`

## Fix

Add an early check for reflexivity at the beginning of both functions:

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -336,6 +336,10 @@ def is_subperiod(source, target) -> bool:
     """
     if target is None or source is None:
         return False
+
+    if source == target:
+        return True
+
     source = _maybe_coerce_freq(source)
     target = _maybe_coerce_freq(target)

@@ -388,6 +392,10 @@ def is_superperiod(source, target) -> bool:
     """
     if target is None or source is None:
         return False
+
+    if source == target:
+        return True
+
     source = _maybe_coerce_freq(source)
     target = _maybe_coerce_freq(target)
```