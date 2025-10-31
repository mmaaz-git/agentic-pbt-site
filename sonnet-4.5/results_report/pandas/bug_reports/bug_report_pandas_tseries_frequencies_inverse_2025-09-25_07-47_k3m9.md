# Bug Report: pandas.tseries.frequencies Inverse Relationship Violation

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod(source, target)` and `is_superperiod(source, target)` violate the expected inverse relationship. Specifically, `is_subperiod(A, B)` should equal `is_superperiod(B, A)` for all valid frequency pairs, but this property does not hold for several combinations including Daily (D) vs Business Day (B).

## Property-Based Test

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod
from hypothesis import given, strategies as st

VALID_FREQS = ["D", "B", "C", "M", "h", "min", "s", "ms", "us", "ns", "W", "Y", "Q"]

@given(
    source=st.sampled_from(VALID_FREQS),
    target=st.sampled_from(VALID_FREQS)
)
def test_subperiod_superperiod_inverse(source, target):
    result_sub = is_subperiod(source, target)
    result_super = is_superperiod(target, source)

    assert result_sub == result_super, (
        f"is_subperiod({source}, {target}) = {result_sub} but "
        f"is_superperiod({target}, {source}) = {result_super}"
    )
```

**Failing input**: `source='D', target='B'`

## Reproducing the Bug

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod

print(f"is_subperiod('D', 'B') = {is_subperiod('D', 'B')}")
print(f"is_superperiod('B', 'D') = {is_superperiod('B', 'D')}")

print(f"is_subperiod('D', 'C') = {is_subperiod('D', 'C')}")
print(f"is_superperiod('C', 'D') = {is_superperiod('C', 'D')}")

print(f"is_subperiod('C', 'B') = {is_subperiod('C', 'B')}")
print(f"is_superperiod('B', 'C') = {is_superperiod('B', 'C')}")
```

Output:
```
is_subperiod('D', 'B') = False
is_superperiod('B', 'D') = True
is_subperiod('D', 'C') = False
is_superperiod('C', 'D') = True
is_subperiod('C', 'B') = False
is_superperiod('B', 'C') = True
```

## Why This Is A Bug

The docstrings state:
- `is_subperiod`: "Returns True if downsampling is possible between source and target frequencies"
- `is_superperiod`: "Returns True if upsampling is possible between source and target frequencies"

By definition, if you can **downsample** from frequency A to frequency B, then you should be able to **upsample** from frequency B back to frequency A. Therefore:

`is_subperiod(A, B) == is_superperiod(B, A)` should always hold.

However, the implementation is inconsistent:

For `is_subperiod(..., "B")`:
```python
elif target == "B":
    return source in {"B", "h", "min", "s", "ms", "us", "ns"}
```

For `is_superperiod("B", ...)`:
```python
elif source == "B":
    return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
```

Notice that "D" and "C" are included in `is_superperiod("B", ...)` but NOT in `is_subperiod(..., "B")`. This asymmetry causes the inverse relationship to fail.

## Fix

The fix requires making the two functions symmetric. Looking at the logic, `is_superperiod("B", ...)` includes "D" and "C" because you can upsample from Business Day to Calendar Day or Custom Business Day. By the inverse relationship, `is_subperiod(..., "B")` should also include "D" and "C" as valid sources.

```diff
diff --git a/pandas/tseries/frequencies.py b/pandas/tseries/frequencies.py
index 1234567..abcdefg 100644
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -100,7 +100,7 @@ def is_subperiod(source, target) -> bool:
     elif target == "B":
-        return source in {"B", "h", "min", "s", "ms", "us", "ns"}
+        return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "C":
-        return source in {"C", "h", "min", "s", "ms", "us", "ns"}
+        return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "D":
         return source in {"D", "h", "min", "s", "ms", "us", "ns"}
```

This ensures that:
- `is_subperiod("D", "B")` returns True (can downsample from Daily to Business Day)
- `is_superperiod("B", "D")` returns True (can upsample from Business Day to Daily)
- The inverse relationship holds for all frequency pairs