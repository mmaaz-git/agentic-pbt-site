# Bug Report: pandas.tseries.frequencies is_subperiod/is_superperiod Inconsistent Behavior

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`is_subperiod(freq, freq)` and `is_superperiod(freq, freq)` return inconsistent results across different frequencies. Some frequencies (D, W, h, min, s) return `True` when comparing a frequency to itself, while others (M, Q) return `False`. Additionally, 'Y' returns `False` for `is_subperiod` but `True` for `is_superperiod` when compared to itself. This violates the semantic expectation that a frequency cannot be its own sub-period or super-period.

## Property-Based Test

```python
import pandas.tseries.frequencies as frequencies
from hypothesis import given, strategies as st

@given(freq1=st.sampled_from(['D', 'W', 'M', 'Q', 'Y']))
def test_is_subperiod_is_not_superperiod_itself(freq1):
    """
    Property: A frequency should not be its own sub/superperiod.

    Evidence: is_subperiod checks if downsampling is possible,
    so a frequency can't be downsampled to itself.
    """
    result = frequencies.is_subperiod(freq1, freq1)
    assert result == False, \
        f"is_subperiod({freq1}, {freq1}) should be False but got {result}"
```

**Failing input**: Any frequency in `['D', 'W', 'h', 'min', 's']`

## Reproducing the Bug

```python
import pandas.tseries.frequencies as frequencies

freqs = ['D', 'W', 'M', 'Q', 'Y', 'h', 'min', 's']

for freq in freqs:
    is_sub = frequencies.is_subperiod(freq, freq)
    is_super = frequencies.is_superperiod(freq, freq)
    print(f'Frequency: {freq:5s}  is_subperiod({freq}, {freq}) = {is_sub}  is_superperiod({freq}, {freq}) = {is_super}')
```

**Output:**
```
Frequency: D      is_subperiod(D, D) = True  is_superperiod(D, D) = True
Frequency: W      is_subperiod(W, W) = True  is_superperiod(W, W) = True
Frequency: M      is_subperiod(M, M) = False  is_superperiod(M, M) = False
Frequency: Q      is_subperiod(Q, Q) = False  is_superperiod(Q, Q) = False
Frequency: Y      is_subperiod(Y, Y) = False  is_superperiod(Y, Y) = True
Frequency: h      is_subperiod(h, h) = True  is_superperiod(h, h) = True
Frequency: min    is_subperiod(min, min) = True  is_superperiod(min, min) = True
Frequency: s      is_subperiod(s, s) = True  is_superperiod(s, s) = True
```

## Why This Is A Bug

The docstrings state:
- `is_subperiod`: "Returns True if downsampling is possible between source and target frequencies"
- `is_superperiod`: "Returns True if upsampling is possible between source and target frequencies"

Logically, you cannot downsample or upsample from a frequency to itself - that is the identity operation, not sampling. The behavior should be consistent across all frequencies, returning `False` when `source == target`.

The inconsistency arises from the implementation: some frequencies explicitly include themselves in the check set, while others don't. For example, in `is_subperiod`:

```python
elif target == "D":
    return source in {"D", "h", "min", "s", "ms", "us", "ns"}  # D includes itself

elif _is_monthly(target):
    return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}  # M does NOT include itself
```

## Fix

The fix is to ensure that when `source == target`, both functions return `False` (or consistently return `True` if that's the intended semantics, though that would contradict the docstrings).

Recommended approach: Add an early check in both functions:

```diff
diff --git a/pandas/tseries/frequencies.py b/pandas/tseries/frequencies.py
index 1234567..abcdefg 100644
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -440,6 +440,8 @@ def is_subperiod(source, target) -> bool:
     """
     if target is None or source is None:
         return False
+    if source == target:
+        return False
     source = _maybe_coerce_freq(source)
     target = _maybe_coerce_freq(target)

@@ -480,6 +482,8 @@ def is_superperiod(source, target) -> bool:
     """
     if target is None or source is None:
         return False
+    if source == target and not _is_annual(source):
+        return False
     source = _maybe_coerce_freq(source)
     target = _maybe_coerce_freq(target)
```

Note: The `is_superperiod` fix is more complex because annual frequencies have special logic for comparing rule months. A simpler alternative would be to remove the frequency from its own check set in each conditional branch.