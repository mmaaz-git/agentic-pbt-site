# Bug Report: pandas.tseries.frequencies Symmetry Violation

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod` and `is_superperiod` are not symmetric inverses as their documentation and semantics suggest. If upsampling is possible from frequency A to frequency B (i.e., `is_superperiod(A, B) == True`), then downsampling should be possible from B to A (i.e., `is_subperiod(B, A) == True`). However, this property is violated for several frequency combinations involving 'D' (Day), 'B' (BusinessDay), and 'C' (CustomBusinessDay).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.tseries.frequencies as frequencies

freq_strings = st.sampled_from([
    "D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns",
    "W", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT", "W-SUN",
    "Q", "Q-JAN", "Q-FEB", "Q-MAR", "Q-APR", "Q-MAY", "Q-JUN",
    "Q-JUL", "Q-AUG", "Q-SEP", "Q-OCT", "Q-NOV", "Q-DEC",
    "A", "A-JAN", "A-FEB", "A-MAR", "A-APR", "A-MAY", "A-JUN",
    "A-JUL", "A-AUG", "A-SEP", "A-OCT", "A-NOV", "A-DEC",
    "BM", "BQ", "BA", "MS", "QS", "AS",
])

@given(source=freq_strings, target=freq_strings)
@settings(max_examples=1000)
def test_subperiod_superperiod_symmetry(source, target):
    sub_result = frequencies.is_subperiod(source, target)
    super_result = frequencies.is_superperiod(target, source)

    assert sub_result == super_result, (
        f"Symmetry broken: is_subperiod({source!r}, {target!r}) = {sub_result}, "
        f"but is_superperiod({target!r}, {source!r}) = {super_result}"
    )
```

**Failing input**: `source='D', target='C'`

## Reproducing the Bug

```python
import pandas.tseries.frequencies as frequencies

print(f"is_subperiod('D', 'C') = {frequencies.is_subperiod('D', 'C')}")
print(f"is_superperiod('C', 'D') = {frequencies.is_superperiod('C', 'D')}")

print(f"is_subperiod('B', 'C') = {frequencies.is_subperiod('B', 'C')}")
print(f"is_superperiod('C', 'B') = {frequencies.is_superperiod('C', 'B')}")

print(f"is_subperiod('D', 'B') = {frequencies.is_subperiod('D', 'B')}")
print(f"is_superperiod('B', 'D') = {frequencies.is_superperiod('B', 'D')}")
```

Output:
```
is_subperiod('D', 'C') = False
is_superperiod('C', 'D') = True
is_subperiod('B', 'C') = False
is_superperiod('C', 'B') = True
is_subperiod('D', 'B') = False
is_superperiod('B', 'D') = True
```

## Why This Is A Bug

The functions `is_subperiod` and `is_superperiod` test whether downsampling and upsampling (respectively) are possible between two frequencies. These operations are inverses of each other: if you can upsample from A to B, then you must be able to downsample from B to A.

However, the current implementation violates this fundamental relationship:
- `is_superperiod('C', 'D')` returns `True`, indicating upsampling from C to D is possible
- `is_subperiod('D', 'C')` returns `False`, indicating downsampling from D to C is not possible

This is logically inconsistent. Looking at the source code:

In `is_subperiod`, when `target == "C"`:
```python
return source in {"C", "h", "min", "s", "ms", "us", "ns"}
```
Note that 'D' and 'B' are NOT in this set.

In `is_superperiod`, when `source == "C"`:
```python
return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
```
Note that 'D' and 'B' ARE in this set.

The same asymmetry exists for 'B' and 'D'. These functions should be symmetric with respect to their arguments.

## Fix

The bug occurs because `is_subperiod` and `is_superperiod` have inconsistent logic for the 'B', 'C', and 'D' frequency codes. The fix is to ensure that if `is_superperiod(source, target)` returns True, then `is_subperiod(target, source)` must also return True.

Looking at the code, the issue is in the `is_subperiod` function. When the target is 'B', 'C', or 'D', the function should include the other day-level frequencies in the acceptable source set.

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -31,11 +31,11 @@ def is_subperiod(source, target) -> bool:
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

This ensures that 'D', 'B', and 'C' are treated consistently, allowing downsampling and upsampling to be symmetric operations.