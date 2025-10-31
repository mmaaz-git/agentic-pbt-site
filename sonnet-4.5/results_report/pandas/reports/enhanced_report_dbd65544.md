# Bug Report: pandas.tseries.frequencies is_subperiod/is_superperiod Inconsistent Self-Comparison Behavior

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod(freq, freq)` and `is_superperiod(freq, freq)` exhibit inconsistent behavior when a frequency is compared to itself. For frequencies like 'D', 'W', 'h', 'min', and 's', both functions return `True`, while for 'M' and 'Q' they return `False`. The 'Y' frequency shows asymmetric behavior, returning `False` for `is_subperiod` but `True` for `is_superperiod`.

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

if __name__ == "__main__":
    test_is_subperiod_is_not_superperiod_itself()
```

<details>

<summary>
**Failing input**: `freq1='D'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 17, in <module>
    test_is_subperiod_is_not_superperiod_itself()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 5, in test_is_subperiod_is_not_superperiod_itself
    def test_is_subperiod_is_not_superperiod_itself(freq1):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 13, in test_is_subperiod_is_not_superperiod_itself
    assert result == False, \
           ^^^^^^^^^^^^^^^
AssertionError: is_subperiod(D, D) should be False but got True
Falsifying example: test_is_subperiod_is_not_superperiod_itself(
    freq1='D',
)
```
</details>

## Reproducing the Bug

```python
import pandas.tseries.frequencies as frequencies

freqs = ['D', 'W', 'M', 'Q', 'Y', 'h', 'min', 's']

for freq in freqs:
    is_sub = frequencies.is_subperiod(freq, freq)
    is_super = frequencies.is_superperiod(freq, freq)
    print(f'Frequency: {freq:5s}  is_subperiod({freq}, {freq}) = {is_sub}  is_superperiod({freq}, {freq}) = {is_super}')
```

<details>

<summary>
Output showing inconsistent self-comparison results
</summary>
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
</details>

## Why This Is A Bug

The docstrings for these functions state:
- `is_subperiod`: "Returns True if downsampling is possible between source and target frequencies"
- `is_superperiod`: "Returns True if upsampling is possible between source and target frequencies"

Semantically, downsampling or upsampling from a frequency to itself is not a sampling operation—it's the identity operation. A frequency cannot be a sub-period or super-period of itself, just as a set cannot be a proper subset of itself.

The inconsistency stems from the implementation details in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/tseries/frequencies.py`:

1. For frequencies like 'D', 'h', 'min', 's' (lines 472, 474, 476, 478), the implementation includes the frequency in its own check set:
   ```python
   elif target == "D":
       return source in {"D", "h", "min", "s", "ms", "us", "ns"}  # "D" includes itself
   ```

2. For monthly and quarterly frequencies (lines 464, 462), the frequency is NOT included in its check set:
   ```python
   elif _is_monthly(target):
       return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}  # "M" not included
   ```

3. For annual frequencies (line 511-512), there's special logic in `is_superperiod` that compares rule months when both are annual, leading to the asymmetric behavior.

## Relevant Context

These functions are part of the public API (included in `__all__` on line 599-600) and are used for determining valid frequency conversions in pandas time series operations. While pandas' `resample()` method does allow resampling to the same frequency (which performs no operation), the semantic meaning of these helper functions suggests they should consistently return False for self-comparisons.

The inconsistency could lead to subtle bugs in user code that relies on these functions to determine valid frequency conversions, especially when building generic time series processing pipelines.

## Proposed Fix

Add an early check in both functions to handle the self-comparison case consistently:

```diff
diff --git a/pandas/tseries/frequencies.py b/pandas/tseries/frequencies.py
index current..fixed
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -450,6 +452,10 @@ def is_subperiod(source, target) -> bool:
     if target is None or source is None:
         return False
+
+    # A frequency cannot be its own sub-period
+    if source == target:
+        return False
+
     source = _maybe_coerce_freq(source)
     target = _maybe_coerce_freq(target)

@@ -505,6 +511,10 @@ def is_superperiod(source, target) -> bool:
     if target is None or source is None:
         return False
+
+    # A frequency cannot be its own super-period
+    if source == target:
+        return False
+
     source = _maybe_coerce_freq(source)
     target = _maybe_coerce_freq(target)
```

Note: The comparison should be done before the `_maybe_coerce_freq` calls to ensure we're comparing the original input values. After coercion, additional logic may be needed for edge cases involving different representations of the same frequency.