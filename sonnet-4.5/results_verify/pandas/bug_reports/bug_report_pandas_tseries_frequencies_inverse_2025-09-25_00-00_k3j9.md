# Bug Report: pandas.tseries.frequencies Inverse Relationship Violation

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod` and `is_superperiod` violate their expected inverse relationship for annual frequencies. When comparing an annual frequency to itself, `is_superperiod(freq, freq)` returns `True` but `is_subperiod(freq, freq)` returns `False`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.tseries.frequencies import is_subperiod, is_superperiod

VALID_FREQUENCIES = [
    "ns", "us", "ms", "s", "min", "h",
    "D", "B", "C",
    "W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT",
    "M", "MS", "BM", "BMS",
    "Q", "QS", "BQ", "BQS",
    "Q-JAN", "Q-FEB", "Q-MAR", "Q-APR", "Q-MAY", "Q-JUN",
    "Q-JUL", "Q-AUG", "Q-SEP", "Q-OCT", "Q-NOV", "Q-DEC",
    "Y", "YS", "BY", "BYS",
    "Y-JAN", "Y-FEB", "Y-MAR", "Y-APR", "Y-MAY", "Y-JUN",
    "Y-JUL", "Y-AUG", "Y-SEP", "Y-OCT", "Y-NOV", "Y-DEC",
]

freq_strategy = st.sampled_from(VALID_FREQUENCIES)

@given(source=freq_strategy, target=freq_strategy)
@settings(max_examples=1000)
def test_inverse_relationship_superperiod_subperiod(source, target):
    """
    Property: If is_superperiod(source, target) is True,
    then is_subperiod(target, source) should also be True.
    """
    super_result = is_superperiod(source, target)
    sub_result = is_subperiod(target, source)

    if super_result:
        assert sub_result, (
            f"is_superperiod({source!r}, {target!r}) = True, "
            f"but is_subperiod({target!r}, {source!r}) = {sub_result}"
        )
```

**Failing input**: `source='Y-JAN', target='Y-JAN'`

## Reproducing the Bug

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod

freq = 'Y-JAN'

super_result = is_superperiod(freq, freq)
sub_result = is_subperiod(freq, freq)

print(f"is_superperiod('{freq}', '{freq}') = {super_result}")
print(f"is_subperiod('{freq}', '{freq}') = {sub_result}")

assert super_result == sub_result, \
    f"Expected both to return the same value, but got {super_result} and {sub_result}"
```

Output:
```
is_superperiod('Y-JAN', 'Y-JAN') = True
is_subperiod('Y-JAN', 'Y-JAN') = False
AssertionError: Expected both to return the same value, but got True and False
```

## Why This Is A Bug

The documentation states that `is_superperiod` returns True if upsampling is possible, and `is_subperiod` returns True if downsampling is possible. These functions should be mathematical inverses: if `is_superperiod(a, b)` is True, then `is_subperiod(b, a)` must also be True.

When comparing a frequency to itself (e.g., 'Y-JAN' to 'Y-JAN'), both upsampling and downsampling should have the same result, as neither operation is actually needed - the frequencies are identical.

The bug affects all annual frequencies: Y, Y-JAN, Y-FEB, Y-MAR, Y-APR, Y-MAY, Y-JUN, Y-JUL, Y-AUG, Y-SEP, Y-OCT, Y-NOV, Y-DEC, YS, BY, BYS.

## Fix

The issue is in the `is_subperiod` function. When the target is an annual frequency, the function should check if the source is also the same annual frequency before falling through to the default check.

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -XXX,6 +XXX,8 @@ def is_subperiod(source, target) -> bool:
     target = _maybe_coerce_freq(target)

     if _is_annual(target):
+        if _is_annual(source):
+            return get_rule_month(source) == get_rule_month(target)
         if _is_quarterly(source):
             return _quarter_months_conform(
                 get_rule_month(source), get_rule_month(target)
```

This change mirrors the logic in `is_superperiod` and ensures that comparing an annual frequency to itself returns `True` for both functions.