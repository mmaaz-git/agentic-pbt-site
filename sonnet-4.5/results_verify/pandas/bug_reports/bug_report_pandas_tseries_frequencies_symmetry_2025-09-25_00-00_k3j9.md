# Bug Report: pandas.tseries.frequencies Symmetry Violation

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_subperiod` and `is_superperiod` functions violate a fundamental symmetry property: `is_subperiod(a, b)` should equal `is_superperiod(b, a)` for all frequency pairs. This property is violated when both frequencies are annual (e.g., 'Y', 'Y-JAN', 'Y-FEB').

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.tseries.frequencies import is_subperiod, is_superperiod

freq_strings = st.sampled_from([
    'D', 'B', 'C', 'h', 'min', 's', 'ms', 'us', 'ns',
    'M', 'BM', 'W', 'Y', 'Q',
    'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN',
    'Q-JAN', 'Q-FEB', 'Q-MAR', 'Q-APR', 'Q-MAY', 'Q-JUN',
    'Q-JUL', 'Q-AUG', 'Q-SEP', 'Q-OCT', 'Q-NOV', 'Q-DEC',
    'Y-JAN', 'Y-FEB', 'Y-MAR', 'Y-APR', 'Y-MAY', 'Y-JUN',
    'Y-JUL', 'Y-AUG', 'Y-SEP', 'Y-OCT', 'Y-NOV', 'Y-DEC',
])

@given(freq_strings, freq_strings)
def test_subperiod_superperiod_symmetry_strings(source, target):
    result_super = is_superperiod(source, target)
    result_sub = is_subperiod(target, source)
    assert result_super == result_sub
```

**Failing input**: `source='Y-JAN', target='Y-JAN'`

## Reproducing the Bug

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod

freq = 'Y-JAN'
print(f"is_superperiod('{freq}', '{freq}') = {is_superperiod(freq, freq)}")
print(f"is_subperiod('{freq}', '{freq}') = {is_subperiod(freq, freq)}")

assert is_superperiod(freq, freq) == is_subperiod(freq, freq)
```

Output:
```
is_superperiod('Y-JAN', 'Y-JAN') = True
is_subperiod('Y-JAN', 'Y-JAN') = False
AssertionError: Symmetry violated!
```

The bug affects all annual frequencies: 'Y', 'Y-JAN', 'Y-FEB', 'Y-MAR', etc.

## Why This Is A Bug

1. **Documented symmetry**: The existing test suite explicitly tests for symmetry between `is_superperiod` and `is_subperiod` (see `test_super_sub_symmetry` in `pandas/tests/tseries/frequencies/test_frequencies.py`).

2. **Mathematical consistency**: If frequency A is a superperiod of frequency B, then by definition, frequency B should be a subperiod of frequency A. This is a fundamental property of the period hierarchy.

3. **Inconsistent reflexivity**: A frequency should be both a superperiod and a subperiod of itself, yet `is_subperiod('Y-JAN', 'Y-JAN')` returns False while `is_superperiod('Y-JAN', 'Y-JAN')` returns True.

## Fix

The bug is in the `is_subperiod` function at lines 455-460. When `target` is an annual frequency, the function doesn't check if `source` is also an annual frequency with the same month anchor. This check exists in `is_superperiod` (lines 511-512) but is missing from `is_subperiod`.

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
```

This fix mirrors the logic in `is_superperiod` and ensures that when both frequencies are annual, they are considered subperiods/superperiods of each other if and only if they have the same month anchor.