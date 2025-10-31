# Bug Report: pandas.tseries.frequencies is_subperiod and is_superperiod Not Inverse

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod` and `is_superperiod` are supposed to be inverse operations, but they are not. Specifically, when `is_superperiod(source, target)` returns `True`, `is_subperiod(target, source)` should also return `True`, but it doesn't for several frequency pairs involving 'D', 'B', and 'C'.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from pandas.tseries.frequencies import is_subperiod, is_superperiod

freq_strings = st.sampled_from([
    "D", "B", "C", "h", "min", "s", "ms", "us", "ns",
    "M", "BM", "W", "Y", "Q",
])

@settings(max_examples=1000)
@given(source=freq_strings, target=freq_strings)
def test_superperiod_subperiod_inverse(source, target):
    if is_superperiod(source, target):
        assert is_subperiod(target, source), (
            f"If {source} is_superperiod of {target}, "
            f"then {target} should be is_subperiod of {source}"
        )
```

**Failing input**: `source='D', target='B'`

## Reproducing the Bug

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod

print(f"is_superperiod('D', 'B') = {is_superperiod('D', 'B')}")
print(f"is_subperiod('B', 'D') = {is_subperiod('B', 'D')}")

print(f"is_superperiod('D', 'C') = {is_superperiod('D', 'C')}")
print(f"is_subperiod('C', 'D') = {is_subperiod('C', 'D')}")

print(f"is_superperiod('B', 'D') = {is_superperiod('B', 'D')}")
print(f"is_subperiod('D', 'B') = {is_subperiod('D', 'B')}")
```

**Output:**
```
is_superperiod('D', 'B') = True
is_subperiod('B', 'D') = False
is_superperiod('D', 'C') = True
is_subperiod('C', 'D') = False
is_superperiod('B', 'D') = True
is_subperiod('D', 'B') = False
```

## Why This Is A Bug

The docstrings indicate these functions should be inverse operations:
- `is_subperiod`: "Returns True if downsampling is possible between source and target frequencies"
- `is_superperiod`: "Returns True if upsampling is possible between source and target frequencies"

If downsampling from frequency A to B is possible, then upsampling from B to A should be possible, and vice versa. However, the current implementation violates this property.

Looking at the code in `frequencies.py`:
- Lines 525-530: `is_superperiod("D"|"B"|"C", target)` returns True when target is in `{"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}`
- Lines 467-472: `is_subperiod(source, "D"|"B"|"C")` only returns True when source is in a restricted set that excludes the other day frequencies

For example:
- `is_superperiod("D", "B")` checks if "B" is in {"D", "C", "B", ...} → True
- `is_subperiod("B", "D")` checks if "B" is in {"D", "h", "min", ...} → False (missing "B" and "C")

## Fix

The bug is in the `is_subperiod` function. Lines 467-472 need to be updated to include all day-level frequencies ("D", "B", "C") when the target is any day-level frequency:

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -465,11 +465,11 @@ def is_subperiod(source, target) -> bool:
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