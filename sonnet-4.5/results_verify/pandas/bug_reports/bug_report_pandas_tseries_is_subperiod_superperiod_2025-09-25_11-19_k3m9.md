# Bug Report: pandas.tseries.frequencies Inconsistent is_subperiod/is_superperiod for Same Frequency

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`is_subperiod` and `is_superperiod` return inconsistent results when source and target are the same frequency. For annual frequencies like 'Y', `is_subperiod('Y', 'Y')` returns False while `is_superperiod('Y', 'Y')` returns True, violating the expected inverse relationship between these functions.

## Property-Based Test

```python
from pandas.tseries import frequencies
from hypothesis import given, strategies as st, settings

freq_strings = ['D', 'W', 'M', 'Q', 'Y', 'h', 'min', 's', 'ms', 'us', 'ns', 'B']

@given(st.sampled_from(freq_strings), st.sampled_from(freq_strings))
@settings(max_examples=200)
def test_subperiod_superperiod_inverse(source_freq, target_freq):
    is_sub = frequencies.is_subperiod(source_freq, target_freq)
    is_super = frequencies.is_superperiod(target_freq, source_freq)
    assert is_sub == is_super, f"is_subperiod({source_freq}, {target_freq}) = {is_sub}, but is_superperiod({target_freq}, {source_freq}) = {is_super}"
```

**Failing input**: `source_freq='Y', target_freq='Y'`

## Reproducing the Bug

```python
from pandas.tseries import frequencies

is_sub = frequencies.is_subperiod('Y', 'Y')
is_super = frequencies.is_superperiod('Y', 'Y')

print(f"is_subperiod('Y', 'Y') = {is_sub}")
print(f"is_superperiod('Y', 'Y') = {is_super}")

assert is_sub == is_super, f"Expected both to be equal, got {is_sub} != {is_super}"
```

## Why This Is A Bug

These functions are documented as inverse operations: `is_subperiod(A, B)` should equal `is_superperiod(B, A)`. When both frequencies are the same, neither downsampling nor upsampling is needed, so the functions should agree on the return value (either both True for identity transformation, or both False for no transformation needed).

The current behavior violates this property for annual frequencies. Looking at the source code in `pandas/tseries/frequencies.py`:

- `is_subperiod('Y', 'Y')`: When target is annual, it checks if source is in `{"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}`, which excludes 'Y', returning False.
- `is_superperiod('Y', 'Y')`: When source is annual and target is annual, it returns `get_rule_month(source) == get_rule_month(target)`, which is True.

## Fix

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -485,6 +485,8 @@ def is_subperiod(source, target) -> bool:
     target = _maybe_coerce_freq(target)

     if _is_annual(target):
+        if _is_annual(source):
+            return get_rule_month(source) == get_rule_month(target)
         if _is_quarterly(source):
             return _quarter_months_conform(
                 get_rule_month(source), get_rule_month(target)
```