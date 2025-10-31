# Bug Report: pandas.tseries.frequencies.is_subperiod Reflexivity Violation

**Target**: `pandas.tseries.frequencies.is_subperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_subperiod` function violates the reflexivity property: for certain frequencies (M, Q, Y, and their variants), `is_subperiod(freq, freq)` returns `False` when it should return `True`. Additionally, this creates an inconsistency with `is_superperiod`, where `is_subperiod(freq, freq)` and `is_superperiod(freq, freq)` return different values for annual frequencies.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.tseries.frequencies import is_subperiod, is_superperiod

FREQ_STRINGS = [
    "ns", "us", "ms", "s", "min", "h",
    "D", "B", "C",
    "W", "M", "Q", "Y",
    "Y-JAN", "Y-FEB", "Q-JAN", "Q-FEB",
]

@given(st.sampled_from(FREQ_STRINGS))
@settings(max_examples=200)
def test_subperiod_reflexivity(freq):
    assert is_subperiod(freq, freq), \
        f"is_subperiod({freq!r}, {freq!r}) should be True (reflexivity)"

@given(st.sampled_from(FREQ_STRINGS), st.sampled_from(FREQ_STRINGS))
@settings(max_examples=1000)
def test_subperiod_superperiod_inverse(source, target):
    result_sub = is_subperiod(source, target)
    result_super = is_superperiod(target, source)
    assert result_sub == result_super, \
        f"is_subperiod({source!r}, {target!r}) = {result_sub}, but is_superperiod({target!r}, {source!r}) = {result_super}"
```

**Failing inputs**:
- `freq='M'` (test_subperiod_reflexivity)
- `freq='Q'` (test_subperiod_reflexivity)
- `freq='Y'` (test_subperiod_reflexivity)
- `source='Y-JAN', target='Y-JAN'` (test_subperiod_superperiod_inverse)

## Reproducing the Bug

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod

assert is_subperiod('M', 'M') == False
assert is_subperiod('Q', 'Q') == False
assert is_subperiod('Y', 'Y') == False
assert is_subperiod('Y-JAN', 'Y-JAN') == False

assert is_superperiod('Y', 'Y') == True
assert is_superperiod('Y-JAN', 'Y-JAN') == True

assert is_subperiod('D', 'D') == True
assert is_subperiod('h', 'h') == True
```

## Why This Is A Bug

The reflexivity property states that any frequency should be considered a subperiod of itself. This is a fundamental mathematical property that users would reasonably expect. The docstring states "Returns True if downsampling is possible between source and target frequencies" - downsampling from a frequency to itself is trivially possible (it's a no-op).

Additionally, the inconsistency between `is_subperiod(freq, freq)` and `is_superperiod(freq, freq)` for annual frequencies violates the documented inverse relationship between these functions.

## Fix

The bug occurs because the sets of valid source frequencies don't include the target frequency itself. The fix is to add the target frequency to each set:

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -503,15 +503,15 @@ def is_subperiod(source, target) -> bool:
             )
         return source in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
     elif _is_quarterly(target):
-        return source in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
+        return source in {target, "D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
     elif _is_monthly(target):
-        return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
+        return source in {target, "D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif _is_weekly(target):
         return source in {target, "D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "B":
-        return source in {"B", "h", "min", "s", "ms", "us", "ns"}
+        return source in {target, "h", "min", "s", "ms", "us", "ns"}
     elif target == "C":
-        return source in {"C", "h", "min", "s", "ms", "us", "ns"}
+        return source in {target, "h", "min", "s", "ms", "us", "ns"}
     elif target == "D":
         return source in {"D", "h", "min", "s", "ms", "us", "ns"}
     elif target == "h":
```

Note: The annual frequency case needs special handling because it already has logic for annual-to-annual comparison in `is_superperiod`. The fix would need to ensure both functions handle this consistently.