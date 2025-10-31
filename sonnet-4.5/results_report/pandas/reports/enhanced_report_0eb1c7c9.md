# Bug Report: pandas.tseries.frequencies Reflexivity Inconsistency for Annual Frequencies

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_subperiod` and `is_superperiod` functions exhibit inconsistent reflexivity behavior when comparing annual frequencies to themselves, returning `True` for `is_superperiod` but `False` for `is_subperiod`, while all other frequency types return consistent values.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for pandas.tseries.frequencies reflexivity bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from pandas.tseries.frequencies import is_subperiod, is_superperiod

freq_strings = st.sampled_from([
    'D', 'h', 'min', 's', 'ms', 'us', 'ns', 'B', 'C',
    'W', 'W-MON', 'M', 'BM', 'Q', 'Q-JAN',
    'Y', 'Y-JAN',
])

@given(freq_strings)
def test_reflexivity_consistency(freq):
    """Test that is_superperiod and is_subperiod have consistent reflexivity."""
    is_super = is_superperiod(freq, freq)
    is_sub = is_subperiod(freq, freq)
    assert is_super == is_sub, f"Inconsistent reflexivity for {freq}: is_superperiod={is_super}, is_subperiod={is_sub}"

if __name__ == "__main__":
    test_reflexivity_consistency()
```

<details>

<summary>
**Failing input**: `'Y'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 24, in <module>
    test_reflexivity_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 17, in test_reflexivity_consistency
    def test_reflexivity_consistency(freq):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 21, in test_reflexivity_consistency
    assert is_super == is_sub, f"Inconsistent reflexivity for {freq}: is_superperiod={is_super}, is_subperiod={is_sub}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Inconsistent reflexivity for Y: is_superperiod=True, is_subperiod=False
Falsifying example: test_reflexivity_consistency(
    freq='Y',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/tseries/frequencies.py:456
        /home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/tseries/frequencies.py:511
        /home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/tseries/frequencies.py:512
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of pandas.tseries.frequencies reflexivity bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.tseries.frequencies import is_subperiod, is_superperiod

# Test various frequency codes
test_freqs = [
    'D', 'h', 'min', 's', 'B', 'C', 'W',
    'M', 'BM', 'Q', 'Q-JAN', 'Y', 'Y-JAN',
]

print("Testing reflexivity: is_superperiod(freq, freq) vs is_subperiod(freq, freq)")
print("=" * 70)
print(f"{'Frequency':<15} {'is_superperiod':<15} {'is_subperiod':<15} {'Consistent?':<15}")
print("-" * 70)

inconsistent_freqs = []

for freq in test_freqs:
    is_super = is_superperiod(freq, freq)
    is_sub = is_subperiod(freq, freq)
    consistent = is_super == is_sub

    if not consistent:
        inconsistent_freqs.append(freq)

    print(f"{freq:<15} {str(is_super):<15} {str(is_sub):<15} {'Yes' if consistent else 'NO':<15}")

if inconsistent_freqs:
    print("\n" + "=" * 70)
    print(f"INCONSISTENT FREQUENCIES FOUND: {', '.join(inconsistent_freqs)}")
    print("=" * 70)

    # Demonstrate the specific issue with annual frequencies
    print("\nDetailed analysis for annual frequency 'Y':")
    print(f"  is_superperiod('Y', 'Y') = {is_superperiod('Y', 'Y')}")
    print(f"  is_subperiod('Y', 'Y') = {is_subperiod('Y', 'Y')}")
    print("\nThis violates the reflexivity property - a frequency should have")
    print("consistent behavior when compared to itself.")
```

<details>

<summary>
Inconsistent reflexivity found for annual frequencies
</summary>
```
Testing reflexivity: is_superperiod(freq, freq) vs is_subperiod(freq, freq)
======================================================================
Frequency       is_superperiod  is_subperiod    Consistent?
----------------------------------------------------------------------
D               True            True            Yes
h               True            True            Yes
min             True            True            Yes
s               True            True            Yes
B               True            True            Yes
C               True            True            Yes
W               True            True            Yes
M               False           False           Yes
BM              False           False           Yes
Q               False           False           Yes
Q-JAN           False           False           Yes
Y               True            False           NO
Y-JAN           True            False           NO

======================================================================
INCONSISTENT FREQUENCIES FOUND: Y, Y-JAN
======================================================================

Detailed analysis for annual frequency 'Y':
  is_superperiod('Y', 'Y') = True
  is_subperiod('Y', 'Y') = False

This violates the reflexivity property - a frequency should have
consistent behavior when compared to itself.
```
</details>

## Why This Is A Bug

This bug violates fundamental properties that users would expect from frequency comparison functions:

1. **Reflexivity Inconsistency**: The mathematical concept of reflexivity states that any element should relate to itself consistently. For frequency comparisons, if we ask "is frequency X a sub-period of itself?" and "is frequency X a super-period of itself?", we should get the same answer. Annual frequencies violate this principle.

2. **Symmetry Violation**: The functions are documented to have a symmetry property where `is_superperiod(a, b)` should equal `is_subperiod(b, a)`. When `a == b` (reflexive case), this means `is_superperiod(a, a)` should equal `is_subperiod(a, a)`. The current implementation violates this for annual frequencies.

3. **Behavioral Inconsistency Across Frequency Types**: The functions handle different frequency types inconsistently:
   - Time-based frequencies (D, h, min, s) and business frequencies (B, C): Both functions return `True` (reflexive)
   - Weekly frequencies (W): Both return `True` (reflexive) - explicitly handled in code
   - Monthly frequencies (M, BM): Both return `False` (not reflexive but consistent)
   - Quarterly frequencies (Q): Both return `False` (not reflexive but consistent)
   - Annual frequencies (Y): `is_superperiod` returns `True`, `is_subperiod` returns `False` (INCONSISTENT)

4. **Practical Impact**: This inconsistency can lead to unexpected behavior in pandas operations that rely on these functions for frequency conversion validation, resampling operations, or period arithmetic.

## Relevant Context

Looking at the source code in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/tseries/frequencies.py`:

- The `is_superperiod` function (lines 510-518) explicitly handles the annual-to-annual case at lines 511-512, returning `True` when comparing annual frequencies with the same month alignment.

- The `is_subperiod` function (lines 455-460) does NOT have a corresponding check for annual-to-annual comparison. It only checks if the source is quarterly (line 456) or falls through to check if it's in the allowed set of higher-frequency periods (line 460).

- Weekly frequencies show the intended pattern: both functions explicitly include the source frequency in their target set (lines 466 and 524), ensuring reflexivity.

- The code shows that reflexivity WAS considered for some cases (annual in `is_superperiod`, weekly in both) but not consistently implemented.

## Proposed Fix

The fix requires adding explicit reflexivity checks in `is_subperiod` for annual, quarterly, and monthly frequencies to match the pattern already established in `is_superperiod`:

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -454,6 +454,9 @@ def is_subperiod(source, target) -> bool:
     target = _maybe_coerce_freq(target)

     if _is_annual(target):
+        if _is_annual(source):
+            return get_rule_month(source) == get_rule_month(target)
+
         if _is_quarterly(source):
             return _quarter_months_conform(
                 get_rule_month(source), get_rule_month(target)
@@ -461,8 +464,14 @@ def is_subperiod(source, target) -> bool:
         return source in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
     elif _is_quarterly(target):
+        if _is_quarterly(source):
+            return source == target
+
         return source in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
     elif _is_monthly(target):
+        if _is_monthly(source):
+            return source == target
+
         return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
```