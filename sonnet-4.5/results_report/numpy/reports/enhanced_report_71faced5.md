# Bug Report: pandas.tseries.frequencies Inverse Property Violation Between is_subperiod and is_superperiod

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod` and `is_superperiod` violate the fundamental mathematical inverse property: when `is_subperiod(A, B)` returns True, `is_superperiod(B, A)` should also return True, and vice versa. This logic error affects multiple common frequency pairs including daily/business day conversions and yearly frequencies.

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

if __name__ == "__main__":
    test_subperiod_superperiod_inverse()
```

<details>

<summary>
**Failing input**: `('Y', 'Y')`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 30, in <module>
    test_subperiod_superperiod_inverse()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 16, in test_subperiod_superperiod_inverse
    source=st.sampled_from(FREQ_STRINGS),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 24, in test_subperiod_superperiod_inverse
    assert is_sub == is_super_reverse, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: is_subperiod('Y', 'Y') = False but is_superperiod('Y', 'Y') = True
Falsifying example: test_subperiod_superperiod_inverse(
    source='Y',
    target='Y',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/tseries/frequencies.py:512
```
</details>

## Reproducing the Bug

```python
import pandas.tseries.frequencies

print("Example 1: Yearly to Yearly")
print(f"is_subperiod('Y', 'Y') = {pandas.tseries.frequencies.is_subperiod('Y', 'Y')}")
print(f"is_superperiod('Y', 'Y') = {pandas.tseries.frequencies.is_superperiod('Y', 'Y')}")
print(f"Inverse property violated: {pandas.tseries.frequencies.is_subperiod('Y', 'Y')} != {pandas.tseries.frequencies.is_superperiod('Y', 'Y')}")

print("\nExample 2: Daily and Business Day")
print(f"is_subperiod('D', 'B') = {pandas.tseries.frequencies.is_subperiod('D', 'B')}")
print(f"is_superperiod('B', 'D') = {pandas.tseries.frequencies.is_superperiod('B', 'D')}")
print(f"Inverse property violated: {pandas.tseries.frequencies.is_subperiod('D', 'B')} != {pandas.tseries.frequencies.is_superperiod('B', 'D')}")

print("\nExample 3: Business Day and Daily")
print(f"is_subperiod('B', 'D') = {pandas.tseries.frequencies.is_subperiod('B', 'D')}")
print(f"is_superperiod('D', 'B') = {pandas.tseries.frequencies.is_superperiod('D', 'B')}")
print(f"Inverse property violated: {pandas.tseries.frequencies.is_subperiod('B', 'D')} != {pandas.tseries.frequencies.is_superperiod('D', 'B')}")

print("\nExample 4: Daily and Custom Business Day")
print(f"is_subperiod('D', 'C') = {pandas.tseries.frequencies.is_subperiod('D', 'C')}")
print(f"is_superperiod('C', 'D') = {pandas.tseries.frequencies.is_superperiod('C', 'D')}")
print(f"Inverse property violated: {pandas.tseries.frequencies.is_subperiod('D', 'C')} != {pandas.tseries.frequencies.is_superperiod('C', 'D')}")

print("\nExample 5: Business Day and Custom Business Day")
print(f"is_subperiod('B', 'C') = {pandas.tseries.frequencies.is_subperiod('B', 'C')}")
print(f"is_superperiod('C', 'B') = {pandas.tseries.frequencies.is_superperiod('C', 'B')}")
print(f"Inverse property violated: {pandas.tseries.frequencies.is_subperiod('B', 'C')} != {pandas.tseries.frequencies.is_superperiod('C', 'B')}")
```

<details>

<summary>
Output showing inverse property violations across multiple frequency pairs
</summary>
```
Example 1: Yearly to Yearly
is_subperiod('Y', 'Y') = False
is_superperiod('Y', 'Y') = True
Inverse property violated: False != True

Example 2: Daily and Business Day
is_subperiod('D', 'B') = False
is_superperiod('B', 'D') = True
Inverse property violated: False != True

Example 3: Business Day and Daily
is_subperiod('B', 'D') = False
is_superperiod('D', 'B') = True
Inverse property violated: False != True

Example 4: Daily and Custom Business Day
is_subperiod('D', 'C') = False
is_superperiod('C', 'D') = True
Inverse property violated: False != True

Example 5: Business Day and Custom Business Day
is_subperiod('B', 'C') = False
is_superperiod('C', 'B') = True
Inverse property violated: False != True
```
</details>

## Why This Is A Bug

The functions `is_subperiod` and `is_superperiod` are documented as checking whether downsampling and upsampling are possible between two frequencies. According to their docstrings:

- `is_subperiod(source, target)`: "Returns True if downsampling is possible between source and target frequencies"
- `is_superperiod(source, target)`: "Returns True if upsampling is possible between source and target frequencies"

By mathematical definition, these operations should be inverses of each other: if you can downsample from frequency A to frequency B, then you must be able to upsample from frequency B back to frequency A. This is a fundamental property that any reasonable implementation should maintain.

The bug occurs because of inconsistent frequency sets in the implementation:

1. **Business Day ('B') inconsistency**: In `is_subperiod` (line 468), when `target == "B"`, it only checks if `source in {"B", "h", "min", "s", "ms", "us", "ns"}`, missing 'D' and 'C'. However, in `is_superperiod` (line 526), when `source == "B"`, it checks if `target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}`, which includes 'D' and 'C'.

2. **Custom Business Day ('C') inconsistency**: Similar issue at lines 470 (is_subperiod) vs 528 (is_superperiod).

3. **Daily ('D') inconsistency**: Similar issue at lines 472 (is_subperiod) vs 530 (is_superperiod).

4. **Annual ('Y') reflexive case**: The `is_subperiod` function doesn't handle the reflexive case where source and target are both annual frequencies with the same month, while `is_superperiod` correctly returns True for this case (lines 511-512).

## Relevant Context

- Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/tseries/frequencies.py`
- Functions affected: `is_subperiod` (lines 434-486) and `is_superperiod` (lines 489-544)
- These functions are critical for pandas time series resampling operations and frequency conversion validation
- The bug affects common business analytics use cases involving daily/business day conversions and yearly aggregations
- pandas documentation: https://pandas.pydata.org/docs/reference/api/pandas.tseries.frequencies.is_subperiod.html

## Proposed Fix

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -454,6 +454,9 @@ def is_subperiod(source, target) -> bool:
     target = _maybe_coerce_freq(target)

     if _is_annual(target):
+        if _is_annual(source):
+            return get_rule_month(source) == get_rule_month(target)
+
         if _is_quarterly(source):
             return _quarter_months_conform(
                 get_rule_month(source), get_rule_month(target)
@@ -466,11 +469,11 @@ def is_subperiod(source, target) -> bool:
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
     elif target == "min":
```