# Bug Report: pandas.tseries.frequencies.is_subperiod Reflexivity Violation

**Target**: `pandas.tseries.frequencies.is_subperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_subperiod` function violates the reflexivity property for monthly, quarterly, and annual frequencies, returning `False` for `is_subperiod(freq, freq)` when it should return `True`. This also creates an inverse relationship inconsistency with `is_superperiod` for annual frequencies.

## Property-Based Test

```python
"""Property-based test demonstrating the is_subperiod reflexivity bug using Hypothesis"""

from hypothesis import given, strategies as st, settings
from pandas.tseries.frequencies import is_subperiod, is_superperiod

# Define test frequency strings
FREQ_STRINGS = [
    "ns", "us", "ms", "s", "min", "h",
    "D", "B", "C",
    "W", "M", "Q", "Y",
    "Y-JAN", "Y-FEB", "Q-JAN", "Q-FEB",
]

@given(st.sampled_from(FREQ_STRINGS))
@settings(max_examples=200)
def test_subperiod_reflexivity(freq):
    """Test that is_subperiod satisfies reflexivity: is_subperiod(x, x) should always be True"""
    assert is_subperiod(freq, freq), \
        f"is_subperiod({freq!r}, {freq!r}) should be True (reflexivity)"

@given(st.sampled_from(FREQ_STRINGS), st.sampled_from(FREQ_STRINGS))
@settings(max_examples=1000)
def test_subperiod_superperiod_inverse(source, target):
    """Test that is_subperiod and is_superperiod have an inverse relationship"""
    result_sub = is_subperiod(source, target)
    result_super = is_superperiod(target, source)
    assert result_sub == result_super, \
        f"is_subperiod({source!r}, {target!r}) = {result_sub}, but is_superperiod({target!r}, {source!r}) = {result_super}"

# Run the tests
if __name__ == "__main__":
    print("Testing reflexivity property of is_subperiod...")
    print("=" * 70)
    try:
        test_subperiod_reflexivity()
        print("✓ All reflexivity tests passed")
    except AssertionError as e:
        print(f"✗ Reflexivity test FAILED")
        print(f"  {e}")

    print("\nTesting inverse relationship between is_subperiod and is_superperiod...")
    print("=" * 70)
    try:
        test_subperiod_superperiod_inverse()
        print("✓ All inverse relationship tests passed")
    except AssertionError as e:
        print(f"✗ Inverse relationship test FAILED")
        print(f"  {e}")
```

<details>

<summary>
**Failing input**: `freq='M'` for reflexivity test, `source='Y', target='Y'` for inverse test
</summary>
```
Testing reflexivity property of is_subperiod...
======================================================================
✗ Reflexivity test FAILED
  is_subperiod('M', 'M') should be True (reflexivity)

Testing inverse relationship between is_subperiod and is_superperiod...
======================================================================
✗ Inverse relationship test FAILED
  is_subperiod('Y', 'Y') = False, but is_superperiod('Y', 'Y') = True
```
</details>

## Reproducing the Bug

```python
"""Minimal reproduction case demonstrating the is_subperiod reflexivity bug"""

from pandas.tseries.frequencies import is_subperiod, is_superperiod

print("Testing reflexivity property (is_subperiod(freq, freq) should always be True):")
print("=" * 70)

# Test frequencies that FAIL reflexivity
failing_freqs = ['M', 'Q', 'Y', 'Y-JAN', 'Y-FEB', 'Q-JAN', 'Q-FEB']
for freq in failing_freqs:
    result = is_subperiod(freq, freq)
    print(f"is_subperiod('{freq}', '{freq}') = {result}  {'❌ WRONG' if not result else '✓'}")

print("\nTest frequencies that PASS reflexivity:")
print("-" * 70)

# Test frequencies that work correctly
passing_freqs = ['D', 'h', 'min', 's', 'ms', 'us', 'ns', 'B', 'C', 'W', 'W-MON', 'W-TUE']
for freq in passing_freqs:
    result = is_subperiod(freq, freq)
    print(f"is_subperiod('{freq}', '{freq}') = {result}  {'✓' if result else '❌ WRONG'}")

print("\nTesting inverse relationship inconsistency:")
print("=" * 70)
print("For annual frequencies, is_subperiod and is_superperiod give different results:")
print("-" * 70)

# Test annual frequencies showing the inconsistency
annual_freqs = ['Y', 'Y-JAN', 'Y-FEB']
for freq in annual_freqs:
    sub_result = is_subperiod(freq, freq)
    super_result = is_superperiod(freq, freq)
    print(f"is_subperiod('{freq}', '{freq}') = {sub_result}")
    print(f"is_superperiod('{freq}', '{freq}') = {super_result}")
    if sub_result != super_result:
        print(f"  ❌ INCONSISTENT: These should be equal!")
    print()

print("For M and Q frequencies, both return False (consistent but wrong):")
print("-" * 70)
for freq in ['M', 'Q']:
    sub_result = is_subperiod(freq, freq)
    super_result = is_superperiod(freq, freq)
    print(f"is_subperiod('{freq}', '{freq}') = {sub_result}")
    print(f"is_superperiod('{freq}', '{freq}') = {super_result}")
    if sub_result == super_result:
        print(f"  Consistent, but both should be True")
    print()
```

<details>

<summary>
Reflexivity violations and inverse relationship inconsistencies
</summary>
```
Testing reflexivity property (is_subperiod(freq, freq) should always be True):
======================================================================
is_subperiod('M', 'M') = False  ❌ WRONG
is_subperiod('Q', 'Q') = False  ❌ WRONG
is_subperiod('Y', 'Y') = False  ❌ WRONG
is_subperiod('Y-JAN', 'Y-JAN') = False  ❌ WRONG
is_subperiod('Y-FEB', 'Y-FEB') = False  ❌ WRONG
is_subperiod('Q-JAN', 'Q-JAN') = False  ❌ WRONG
is_subperiod('Q-FEB', 'Q-FEB') = False  ❌ WRONG

Test frequencies that PASS reflexivity:
----------------------------------------------------------------------
is_subperiod('D', 'D') = True  ✓
is_subperiod('h', 'h') = True  ✓
is_subperiod('min', 'min') = True  ✓
is_subperiod('s', 's') = True  ✓
is_subperiod('ms', 'ms') = True  ✓
is_subperiod('us', 'us') = True  ✓
is_subperiod('ns', 'ns') = True  ✓
is_subperiod('B', 'B') = True  ✓
is_subperiod('C', 'C') = True  ✓
is_subperiod('W', 'W') = True  ✓
is_subperiod('W-MON', 'W-MON') = True  ✓
is_subperiod('W-TUE', 'W-TUE') = True  ✓

Testing inverse relationship inconsistency:
======================================================================
For annual frequencies, is_subperiod and is_superperiod give different results:
----------------------------------------------------------------------
is_subperiod('Y', 'Y') = False
is_superperiod('Y', 'Y') = True
  ❌ INCONSISTENT: These should be equal!

is_subperiod('Y-JAN', 'Y-JAN') = False
is_superperiod('Y-JAN', 'Y-JAN') = True
  ❌ INCONSISTENT: These should be equal!

is_subperiod('Y-FEB', 'Y-FEB') = False
is_superperiod('Y-FEB', 'Y-FEB') = True
  ❌ INCONSISTENT: These should be equal!

For M and Q frequencies, both return False (consistent but wrong):
----------------------------------------------------------------------
is_subperiod('M', 'M') = False
is_superperiod('M', 'M') = False
  Consistent, but both should be True

is_subperiod('Q', 'Q') = False
is_superperiod('Q', 'Q') = False
  Consistent, but both should be True
```
</details>

## Why This Is A Bug

This violates the mathematical reflexivity property that any frequency should be considered a subperiod of itself. The function's docstring states it returns `True` if "downsampling is possible between source and target frequencies" - downsampling from a frequency to itself is trivially possible as a no-op operation.

The bug manifests in three critical ways:

1. **Inconsistent behavior across frequency types**: The function correctly returns `True` for `is_subperiod(freq, freq)` for 12 frequency types (D, W, h, min, s, ms, us, ns, B, C, and their variants) but incorrectly returns `False` for M, Q, Y and their variants. This inconsistency cannot be intentional design.

2. **Violates documented inverse relationship**: For annual frequencies, `is_subperiod('Y', 'Y')` returns `False` while `is_superperiod('Y', 'Y')` returns `True`. The documentation implies these functions should have an inverse relationship, making this a clear contract violation.

3. **Contradicts maintainer expectations**: The pandas maintainers have already acknowledged this as a bug in GitHub issue #18553, with a maintainer explicitly stating that "Downsampling at the same frequency should vacuously be true."

## Relevant Context

- **Source code location**: `/pandas/tseries/frequencies.py`, lines 434-486 for `is_subperiod`, lines 489-544 for `is_superperiod`
- **GitHub issue**: https://github.com/pandas-dev/pandas/issues/18553 (open since 2017, labeled as "Bug" by maintainers)
- **Impact**: This is an internal utility function primarily used for resampling operations. While the bug exists, pandas resampling likely has workarounds that prevent user-facing issues.
- **Pattern**: The bug occurs because when checking if a frequency is a subperiod of itself, the code doesn't include the target frequency in the set of valid source frequencies for M, Q, and Y frequencies, while it correctly does so for W frequencies.

## Proposed Fix

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -455,15 +455,22 @@ def is_subperiod(source, target) -> bool:
     if _is_annual(target):
         if _is_quarterly(source):
             return _quarter_months_conform(
                 get_rule_month(source), get_rule_month(target)
             )
+        # Handle reflexivity for annual frequencies
+        if _is_annual(source):
+            return get_rule_month(source) == get_rule_month(target)
         return source in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
     elif _is_quarterly(target):
-        return source in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
+        # Include target frequency for reflexivity
+        if source == target or (_is_quarterly(source) and get_rule_month(source) == get_rule_month(target)):
+            return True
+        return source in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
     elif _is_monthly(target):
+        # Include 'M' for reflexivity
+        if source == target:
+            return True
         return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif _is_weekly(target):
         return source in {target, "D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "B":
         return source in {"B", "h", "min", "s", "ms", "us", "ns"}
```