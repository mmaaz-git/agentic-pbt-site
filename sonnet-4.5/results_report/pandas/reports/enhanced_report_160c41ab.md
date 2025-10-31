# Bug Report: pandas.tseries.frequencies Reflexivity Violation

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod(X, X)` and `is_superperiod(X, X)` violate the mathematical reflexivity property by returning False for monthly ('M'), quarterly ('Q'), and annual ('Y') frequencies when they should always return True.

## Property-Based Test

```python
import pandas.tseries.frequencies as freq
from hypothesis import given, strategies as st, settings

VALID_FREQS = ["D", "h", "B", "C", "M", "W", "Q", "Y", "min", "s", "ms", "us", "ns"]

@given(st.sampled_from(VALID_FREQS))
@settings(max_examples=50)
def test_subperiod_reflexive(freq_str):
    """
    Property: Reflexivity - is_subperiod(X, X) should always be True
    (a frequency is a subperiod of itself).
    """
    result = freq.is_subperiod(freq_str, freq_str)
    assert result, f"Reflexivity violated: is_subperiod({freq_str}, {freq_str}) returned False"

@given(st.sampled_from(VALID_FREQS))
@settings(max_examples=50)
def test_superperiod_reflexive(freq_str):
    """
    Property: Reflexivity - is_superperiod(X, X) should always be True
    (a frequency is a superperiod of itself).
    """
    result = freq.is_superperiod(freq_str, freq_str)
    assert result, f"Reflexivity violated: is_superperiod({freq_str}, {freq_str}) returned False"

# Run the tests
if __name__ == "__main__":
    print("Running property-based tests for is_subperiod reflexivity...")
    try:
        test_subperiod_reflexive()
        print("✓ test_subperiod_reflexive passed")
    except AssertionError as e:
        print(f"✗ test_subperiod_reflexive failed: {e}")

    print("\nRunning property-based tests for is_superperiod reflexivity...")
    try:
        test_superperiod_reflexive()
        print("✓ test_superperiod_reflexive passed")
    except AssertionError as e:
        print(f"✗ test_superperiod_reflexive failed: {e}")
```

<details>

<summary>
**Failing input**: `freq_str='M'`
</summary>
```
Running property-based tests for is_subperiod reflexivity...
✗ test_subperiod_reflexive failed: Reflexivity violated: is_subperiod(M, M) returned False

Running property-based tests for is_superperiod reflexivity...
✗ test_superperiod_reflexive failed: Reflexivity violated: is_superperiod(M, M) returned False
```
</details>

## Reproducing the Bug

```python
import pandas.tseries.frequencies as freq

# Test reflexivity for all frequency types
test_freqs = ["D", "h", "B", "C", "M", "W", "Q", "Y", "min", "s", "ms", "us", "ns"]

print("Testing reflexivity property: is_subperiod(X, X) and is_superperiod(X, X) should always return True")
print("=" * 80)

for f in test_freqs:
    sub_result = freq.is_subperiod(f, f)
    super_result = freq.is_superperiod(f, f)

    if not sub_result or not super_result:
        print(f"FAIL - Frequency: '{f}'")
        print(f"  is_subperiod('{f}', '{f}') = {sub_result} (Expected: True)")
        print(f"  is_superperiod('{f}', '{f}') = {super_result} (Expected: True)")
    else:
        print(f"OK   - Frequency: '{f}' - Both functions correctly return True")
    print()

print("=" * 80)
print("SUMMARY:")
print("Frequencies where reflexivity is violated (bugs):")
failing_sub = [f for f in test_freqs if not freq.is_subperiod(f, f)]
failing_super = [f for f in test_freqs if not freq.is_superperiod(f, f)]
print(f"  is_subperiod reflexivity failures: {', '.join(failing_sub) if failing_sub else 'None'}")
print(f"  is_superperiod reflexivity failures: {', '.join(failing_super) if failing_super else 'None'}")
```

<details>

<summary>
Reflexivity violations detected for M, Q, and Y frequencies
</summary>
```
Testing reflexivity property: is_subperiod(X, X) and is_superperiod(X, X) should always return True
================================================================================
OK   - Frequency: 'D' - Both functions correctly return True

OK   - Frequency: 'h' - Both functions correctly return True

OK   - Frequency: 'B' - Both functions correctly return True

OK   - Frequency: 'C' - Both functions correctly return True

FAIL - Frequency: 'M'
  is_subperiod('M', 'M') = False (Expected: True)
  is_superperiod('M', 'M') = False (Expected: True)

OK   - Frequency: 'W' - Both functions correctly return True

FAIL - Frequency: 'Q'
  is_subperiod('Q', 'Q') = False (Expected: True)
  is_superperiod('Q', 'Q') = False (Expected: True)

FAIL - Frequency: 'Y'
  is_subperiod('Y', 'Y') = False (Expected: True)
  is_superperiod('Y', 'Y') = True (Expected: True)

OK   - Frequency: 'min' - Both functions correctly return True

OK   - Frequency: 's' - Both functions correctly return True

OK   - Frequency: 'ms' - Both functions correctly return True

OK   - Frequency: 'us' - Both functions correctly return True

OK   - Frequency: 'ns' - Both functions correctly return True

================================================================================
SUMMARY:
Frequencies where reflexivity is violated (bugs):
  is_subperiod reflexivity failures: M, Q, Y
  is_superperiod reflexivity failures: M, Q
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical property of reflexivity. By definition, any frequency should be considered both a subperiod and superperiod of itself, as converting from a frequency to itself requires no resampling operation.

The bug occurs due to faulty logic in the implementation. When checking if a frequency is a sub/superperiod of itself, the code fails to include the frequency in its own valid conversion set:

1. **For Monthly ('M')**: In `is_subperiod`, when `_is_monthly(target)` is True (line 463), it checks if source is in `{"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}` (line 464), which excludes 'M' itself. The same issue occurs in `is_superperiod` (line 521-522).

2. **For Quarterly ('Q')**: In `is_subperiod`, when `_is_quarterly(target)` is True (line 461), it checks if source is in `{"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}` (line 462), which excludes 'Q' itself. The same issue occurs in `is_superperiod` (line 519-520).

3. **For Annual ('Y')**: In `is_subperiod`, when `_is_annual(target)` is True (line 455), it first checks if source is quarterly (line 456), and if not, checks if source is in a set that excludes 'Y' (line 460). However, `is_superperiod` correctly handles the reflexive case for annual frequencies by checking if both source and target are annual and have the same rule month (lines 510-512).

The inconsistency is particularly problematic:
- 10 out of 13 frequency types correctly return True for reflexivity
- 3 common frequency types (M, Q, Y) fail, creating unexpected behavior
- The functions' documentation mentions "downsampling" and "upsampling" but doesn't explicitly state that reflexivity should fail

## Relevant Context

- This bug has been previously reported in pandas GitHub Issue #18553, where it was acknowledged and labeled as "Bug"
- The issue affects commonly used frequency types in financial and business time series analysis
- The current behavior is inconsistent: some frequencies (D, h, B, C, W, min, s, ms, us, ns) correctly handle reflexivity while others don't
- Source code location: `/pandas/tseries/frequencies.py` lines 434-486 (`is_subperiod`) and 489-544 (`is_superperiod`)
- The functions are used internally in pandas for resampling operations and frequency conversion logic

## Proposed Fix

Add an early reflexivity check at the beginning of both functions before any other logic:

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -450,6 +450,10 @@ def is_subperiod(source, target) -> bool:
     if target is None or source is None:
         return False
+
+    # Handle reflexivity: a frequency is always a subperiod of itself
+    if source == target:
+        return True
+
     source = _maybe_coerce_freq(source)
     target = _maybe_coerce_freq(target)

@@ -505,6 +509,10 @@ def is_superperiod(source, target) -> bool:
     if target is None or source is None:
         return False
+
+    # Handle reflexivity: a frequency is always a superperiod of itself
+    if source == target:
+        return True
+
     source = _maybe_coerce_freq(source)
     target = _maybe_coerce_freq(target)
```