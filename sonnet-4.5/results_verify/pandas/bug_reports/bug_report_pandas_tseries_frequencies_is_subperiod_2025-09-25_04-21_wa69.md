# Bug Report: pandas.tseries.frequencies Symmetry Violation in is_subperiod/is_superperiod

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod` and `is_superperiod` violate a fundamental symmetry property: when `source == target == 'Y'` (annual frequency), `is_subperiod('Y', 'Y')` returns `False` but `is_superperiod('Y', 'Y')` returns `True`. These functions should be symmetric - if downsampling from A to B is possible, then upsampling from B to A should also be possible.

## Property-Based Test

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod
from hypothesis import given, strategies as st, settings

VALID_FREQ_STRINGS = [
    "Y", "Q", "M", "W", "D", "B", "C", "h", "min", "s", "ms", "us", "ns",
    "Y-JAN", "Y-FEB", "Y-MAR", "Q-JAN", "Q-FEB", "W-MON", "W-TUE"
]

@given(
    source=st.sampled_from(VALID_FREQ_STRINGS),
    target=st.sampled_from(VALID_FREQ_STRINGS)
)
@settings(max_examples=1000)
def test_subperiod_superperiod_symmetry(source, target):
    sub_result = is_subperiod(source, target)
    super_result = is_superperiod(target, source)

    assert sub_result == super_result, (
        f"Symmetry violated: is_subperiod({source!r}, {target!r}) = {sub_result}, "
        f"but is_superperiod({target!r}, {source!r}) = {super_result}"
    )
```

**Failing input**: `source='Y'`, `target='Y'`

## Reproducing the Bug

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod

source = 'Y'
target = 'Y'

sub_result = is_subperiod(source, target)
super_result = is_superperiod(target, source)

print(f"is_subperiod('Y', 'Y') = {sub_result}")
print(f"is_superperiod('Y', 'Y') = {super_result}")

assert sub_result == super_result, "Symmetry violated!"
```

Output:
```
is_subperiod('Y', 'Y') = False
is_superperiod('Y', 'Y') = True
AssertionError: Symmetry violated!
```

## Why This Is A Bug

The functions `is_subperiod` and `is_superperiod` represent inverse operations (downsampling vs upsampling). By definition, these operations should be symmetric: if you can downsample from frequency A to frequency B, then you should be able to upsample from frequency B to frequency A.

When checking if a frequency can be down/upsampled to itself, both functions should return the same result. For annual frequency 'Y', this symmetry is violated.

**Root cause**: In `is_subperiod` (line 455-460), when the target is annual, the code checks if the source is quarterly, but doesn't check if the source is also annual before falling through to a set membership test that excludes 'Y'. In contrast, `is_superperiod` (line 510-518) correctly handles the case where both source and target are annual.

## Fix

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -453,6 +453,9 @@ def is_subperiod(source, target) -> bool:
     target = _maybe_coerce_freq(target)

     if _is_annual(target):
+        if _is_annual(source):
+            return get_rule_month(source) == get_rule_month(target)
+
         if _is_quarterly(source):
             return _quarter_months_conform(
                 get_rule_month(source), get_rule_month(target)
```

This fix mirrors the logic in `is_superperiod` and ensures that when both source and target are annual frequencies, they are compared by their rule months (returning `True` for 'Y' == 'Y').