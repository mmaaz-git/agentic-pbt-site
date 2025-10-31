# Bug Report: pandas.tseries.frequencies Inverse Property Violation

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod` and `is_superperiod` violate the fundamental inverse property: when `is_subperiod(A, B)` returns True, `is_superperiod(B, A)` should also return True (and vice versa). However, multiple frequency pairs violate this property, including daily ('D'), business day ('B'), custom business day ('C'), and yearly ('Y') frequencies.

## Property-Based Test

```python
import pandas.tseries.frequencies
from hypothesis import given, strategies as st, settings

FREQ_STRINGS = [
    "ns", "us", "ms", "s", "min", "h",
    "D", "B", "C",
    "W", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT", "W-SUN",
    "M", "MS", "ME", "BM", "BMS",
    "Q", "QS", "Q-JAN", "Q-FEB", "Q-MAR", "Q-APR", "Q-MAY", "Q-JUN",
    "Q-JUL", "Q-AUG", "Q-SEP", "Q-OCT", "Q-NOV", "Q-DEC",
    "Y", "YS", "Y-JAN", "Y-FEB", "Y-MAR", "Y-APR", "Y-MAY", "Y-JUN",
    "Y-JUL", "Y-AUG", "Y-SEP", "Y-OCT", "Y-NOV", "Y-DEC",
]

@given(
    source=st.sampled_from(FREQ_STRINGS),
    target=st.sampled_from(FREQ_STRINGS)
)
@settings(max_examples=1000)
def test_subperiod_superperiod_inverse(source, target):
    is_sub = pandas.tseries.frequencies.is_subperiod(source, target)
    is_super_reverse = pandas.tseries.frequencies.is_superperiod(target, source)

    assert is_sub == is_super_reverse, (
        f"is_subperiod('{source}', '{target}') = {is_sub} "
        f"but is_superperiod('{target}', '{source}') = {is_super_reverse}"
    )
```

**Failing inputs**:
- `('Y', 'Y')` - yearly to yearly
- `('D', 'B')` and `('B', 'D')` - daily and business day
- `('D', 'C')` and `('C', 'D')` - daily and custom business day
- `('B', 'C')` and `('C', 'B')` - business day and custom business day

## Reproducing the Bug

```python
import pandas.tseries.frequencies

print("Example 1: Yearly to Yearly")
print(f"is_subperiod('Y', 'Y') = {pandas.tseries.frequencies.is_subperiod('Y', 'Y')}")
print(f"is_superperiod('Y', 'Y') = {pandas.tseries.frequencies.is_superperiod('Y', 'Y')}")

print("\nExample 2: Daily and Business Day")
print(f"is_subperiod('D', 'B') = {pandas.tseries.frequencies.is_subperiod('D', 'B')}")
print(f"is_superperiod('B', 'D') = {pandas.tseries.frequencies.is_superperiod('B', 'D')}")

print("\nExample 3: Business Day and Daily")
print(f"is_subperiod('B', 'D') = {pandas.tseries.frequencies.is_subperiod('B', 'D')}")
print(f"is_superperiod('D', 'B') = {pandas.tseries.frequencies.is_superperiod('D', 'B')}")
```

## Why This Is A Bug

The functions `is_subperiod` and `is_superperiod` are documented as checking whether downsampling and upsampling (respectively) are possible between two frequencies. By definition, these should be inverse operations: if you can downsample from A to B, then you should be able to upsample from B to A.

This property is violated because the frequency sets in the two functions are inconsistent:

- In `is_subperiod`, when `target == "B"`, it checks if `source in {"B", "h", "min", "s", "ms", "us", "ns"}` (missing 'D' and 'C')
- In `is_superperiod`, when `source == "B"`, it checks if `target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}` (includes 'D' and 'C')

Similarly for 'D' and 'C', and for yearly frequencies where `is_subperiod` fails to handle the reflexive case.

## Fix

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -xxx,xx +xxx,xx @@ def is_subperiod(source, target) -> bool:
     elif _is_weekly(target):
         return source in {target, "D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "B":
-        return source in {"B", "h", "min", "s", "ms", "us", "ns"}
+        return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "C":
-        return source in {"C", "h", "min", "s", "ms", "us", "ns"}
+        return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "D":
-        return source in {"D", "h", "min", "s", "ms", "us", "ns"}
+        return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "h":
         return source in {"h", "min", "s", "ms", "us", "ns"}
@@ -xxx,xx +xxx,xx @@ def is_subperiod(source, target) -> bool:
     elif target == "ns":
         return source in {"ns"}
+    elif _is_annual(target):
+        return _is_annual(source) and get_rule_month(source) == get_rule_month(target)
     else:
         return False
```