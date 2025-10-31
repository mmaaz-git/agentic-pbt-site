# Bug Report: pandas.tseries.frequencies Inverse Relationship Violation

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod(source, target)` and `is_superperiod(target, source)` should be mathematical inverses, but they return inconsistent results for certain frequency pairs involving Day ('D'), Business Day ('B'), and Custom Business Day ('C').

## Property-Based Test

```python
import pandas.tseries.frequencies as freq
from hypothesis import given, strategies as st, settings

VALID_FREQS = ["D", "h", "B", "C", "M", "W", "Q", "Y", "min", "s", "ms", "us", "ns"]

@given(st.sampled_from(VALID_FREQS), st.sampled_from(VALID_FREQS))
@settings(max_examples=200)
def test_subperiod_superperiod_inverse(source, target):
    """
    Property: is_subperiod and is_superperiod should be inverse relations.
    If is_subperiod(A, B) is True, then is_superperiod(B, A) should also be True.
    """
    sub_result = freq.is_subperiod(source, target)
    super_result = freq.is_superperiod(target, source)

    assert sub_result == super_result, \
        f"Inverse property violated: is_subperiod({source}, {target})={sub_result} but is_superperiod({target}, {source})={super_result}"
```

**Failing input**: `source='D', target='B'`

## Reproducing the Bug

```python
import pandas.tseries.frequencies as freq

print(f"is_subperiod('D', 'B') = {freq.is_subperiod('D', 'B')}")
print(f"is_superperiod('B', 'D') = {freq.is_superperiod('B', 'D')}")

print(f"is_subperiod('D', 'C') = {freq.is_subperiod('D', 'C')}")
print(f"is_superperiod('C', 'D') = {freq.is_superperiod('C', 'D')}")

print(f"is_subperiod('B', 'C') = {freq.is_subperiod('B', 'C')}")
print(f"is_superperiod('C', 'B') = {freq.is_superperiod('C', 'B')}")
```

Output:
```
is_subperiod('D', 'B') = False
is_superperiod('B', 'D') = True
is_subperiod('D', 'C') = False
is_superperiod('C', 'D') = True
is_subperiod('B', 'C') = False
is_superperiod('C', 'B') = True
```

## Why This Is A Bug

The docstrings describe these as inverse operations:
- `is_subperiod`: "Returns True if downsampling is possible between source and target frequencies"
- `is_superperiod`: "Returns True if upsampling is possible between source and target frequencies"

By definition, if downsampling from A to B is possible, then upsampling from B to A should also be possible. The inverse relationship `is_subperiod(A, B) == is_superperiod(B, A)` is a fundamental mathematical property that must hold.

The bug occurs because:
1. In `is_subperiod`, when `target == "B"`, it checks `source in {"B", "h", "min", "s", "ms", "us", "ns"}` (missing "D" and "C")
2. In `is_superperiod`, when `source == "B"`, it checks `target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}` (includes "D" and "C")

This asymmetry violates the inverse relationship.

## Fix

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -355,9 +355,9 @@ def is_subperiod(source, target) -> bool:
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
```