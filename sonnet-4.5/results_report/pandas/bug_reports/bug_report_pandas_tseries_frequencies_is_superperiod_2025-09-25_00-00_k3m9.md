# Bug Report: pandas.tseries.frequencies is_superperiod/is_subperiod Antisymmetry Violation

**Target**: `pandas.tseries.frequencies.is_superperiod` and `pandas.tseries.frequencies.is_subperiod`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_superperiod` function violates the antisymmetric property: both `is_superperiod('D', 'B')` and `is_superperiod('B', 'D')` return `True`, which is logically impossible since two frequencies cannot both be superperiods of each other.

## Property-Based Test

```python
import pandas.tseries.frequencies as freq
from hypothesis import given, strategies as st, settings


freq_strings = st.sampled_from([
    'D', 'h', 'min', 's', 'ms', 'us', 'ns',
    'W', 'B', 'ME', 'QE', 'YE', 'BME', 'BQE', 'BYE',
])


@given(source=freq_strings, target=freq_strings)
@settings(max_examples=500)
def test_is_superperiod_subperiod_inverse(source, target):
    if freq.is_superperiod(source, target):
        assert freq.is_subperiod(target, source), \
            f"is_superperiod({source!r}, {target!r}) is True but is_subperiod({target!r}, {source!r}) is False"
```

**Failing input**: `source='D'`, `target='B'`

## Reproducing the Bug

```python
import pandas.tseries.frequencies as freq

result1 = freq.is_superperiod('D', 'B')
result2 = freq.is_superperiod('B', 'D')

print(f"is_superperiod('D', 'B') = {result1}")
print(f"is_superperiod('B', 'D') = {result2}")

assert result1 and result2
print("BUG: Both return True!")
```

## Why This Is A Bug

The `is_superperiod` function should satisfy the antisymmetric property: if `is_superperiod(A, B)` is `True`, then `is_superperiod(B, A)` must be `False` (for A ≠ B). This is a fundamental requirement for any "super/sub" relationship.

Additionally, `is_superperiod` and `is_subperiod` should be inverse operations: if `is_superperiod(A, B)` is `True`, then `is_subperiod(B, A)` should also be `True`.

Currently, for 'D' (Day) and 'B' (BusinessDay):
- `is_superperiod('D', 'B')` returns `True`
- `is_superperiod('B', 'D')` returns `True` ❌ (violates antisymmetry)
- `is_subperiod('B', 'D')` returns `False` ❌ (violates inverse relationship)
- `is_subperiod('D', 'B')` returns `False` ❌ (violates inverse relationship)

## Fix

The bug is in the `is_superperiod` function. Looking at the source code:

```python
elif source == "B":
    return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
```

and

```python
elif source == "D":
    return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
```

The issue is that 'B' includes 'D' in its target set, and 'D' includes 'B' in its target set, creating a symmetric relationship.

Since Day (D) and BusinessDay (B) represent different calendar systems (D includes weekends, B does not), they should not have a sub/super relationship with each other. The fix is to remove each from the other's target set:

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -xxx,7 +xxx,7 @@ def is_superperiod(source, target) -> bool:
     elif _is_weekly(source):
         return target in {source, "D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif source == "B":
-        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
+        return target in {"B", "h", "min", "s", "ms", "us", "ns"}
     elif source == "C":
-        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
+        return target in {"C", "h", "min", "s", "ms", "us", "ns"}
     elif source == "D":
-        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
+        return target in {"D", "h", "min", "s", "ms", "us", "ns"}
```

Similarly, the same fix should be verified for `is_subperiod` to ensure consistency.