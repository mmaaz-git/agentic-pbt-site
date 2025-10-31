# Bug Report: pandas.tseries.frequencies is_subperiod/is_superperiod Asymmetry

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_subperiod` and `is_superperiod` functions claim to be inverse operations (downsampling vs upsampling), but they return inconsistent results when both frequencies are the same annual frequency (e.g., 'Y', 'Y-JAN', etc.). Specifically, `is_subperiod('Y', 'Y')` returns `False` while `is_superperiod('Y', 'Y')` returns `True`, violating the expected inverse relationship.

## Property-Based Test

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod
from hypothesis import given, strategies as st

FREQ_STRINGS = ['D', 'B', 'C', 'W', 'M', 'Q', 'Y', 'h', 'min', 's', 'ms', 'us', 'ns']

@given(
    source=st.sampled_from(FREQ_STRINGS),
    target=st.sampled_from(FREQ_STRINGS)
)
def test_subperiod_superperiod_inverse(source, target):
    sub_result = is_subperiod(source, target)
    super_result = is_superperiod(target, source)

    assert sub_result == super_result, (
        f"Inverse relationship violated: "
        f"is_subperiod({source}, {target}) = {sub_result}, "
        f"but is_superperiod({target}, {source}) = {super_result}"
    )
```

**Failing input**: `source='Y'`, `target='Y'`

## Reproducing the Bug

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod

freq = 'Y'

sub_result = is_subperiod(freq, freq)
super_result = is_superperiod(freq, freq)

print(f"is_subperiod('{freq}', '{freq}') = {sub_result}")
print(f"is_superperiod('{freq}', '{freq}') = {super_result}")

assert sub_result == super_result
```

Output:
```
is_subperiod('Y', 'Y') = False
is_superperiod('Y', 'Y') = True
AssertionError
```

## Why This Is A Bug

The functions are documented as inverse operations:
- `is_subperiod(source, target)`: "Returns True if **downsampling** is possible between source and target frequencies"
- `is_superperiod(source, target)`: "Returns True if **upsampling** is possible between source and target frequencies"

If downsampling from A to B is possible, then upsampling from B to A should also be possible. This means:
```
is_subperiod(A, B) == is_superperiod(B, A)
```

However, for annual frequencies, this property is violated:
- `is_subperiod('Y', 'Y')` returns `False` (line 455-460 in frequencies.py)
- `is_superperiod('Y', 'Y')` returns `True` (line 510-512 in frequencies.py)

Additionally, the pattern established by other frequency types suggests a frequency should be considered a subperiod of itself:
- Line 466: Weekly includes `{target, ...}` - the target itself
- Line 471: Daily includes `{"D", ...}` - 'D' itself
- But line 460: Annual does NOT include annual frequencies in the allowed set

## Fix

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -454,6 +454,8 @@ def is_subperiod(source, target) -> bool:

     if _is_annual(target):
+        if _is_annual(source):
+            return get_rule_month(source) == get_rule_month(target)
         if _is_quarterly(source):
             return _quarter_months_conform(
                 get_rule_month(source), get_rule_month(target)
```

This makes `is_subperiod` check annual-to-annual frequency compatibility the same way `is_superperiod` does (line 511-512), ensuring the functions remain inverses of each other.