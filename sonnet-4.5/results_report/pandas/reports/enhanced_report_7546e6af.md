# Bug Report: pandas.tseries.frequencies Symmetry Violation

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_subperiod` and `is_superperiod` functions violate the fundamental symmetry property that `is_subperiod(a, b)` should equal `is_superperiod(b, a)` for all valid frequency pairs, specifically failing for annual frequencies.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.tseries.frequencies import is_subperiod, is_superperiod

# Define all valid frequency strings for testing
freq_strings = st.sampled_from([
    'D', 'B', 'C', 'h', 'min', 's', 'ms', 'us', 'ns',
    'M', 'BM', 'W', 'Y', 'Q',
    'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN',
    'Q-JAN', 'Q-FEB', 'Q-MAR', 'Q-APR', 'Q-MAY', 'Q-JUN',
    'Q-JUL', 'Q-AUG', 'Q-SEP', 'Q-OCT', 'Q-NOV', 'Q-DEC',
    'Y-JAN', 'Y-FEB', 'Y-MAR', 'Y-APR', 'Y-MAY', 'Y-JUN',
    'Y-JUL', 'Y-AUG', 'Y-SEP', 'Y-OCT', 'Y-NOV', 'Y-DEC',
])

@given(freq_strings, freq_strings)
def test_subperiod_superperiod_symmetry_strings(source, target):
    """Test that is_superperiod(a, b) == is_subperiod(b, a) for all frequency pairs."""
    result_super = is_superperiod(source, target)
    result_sub = is_subperiod(target, source)
    assert result_super == result_sub, (
        f"Symmetry violated for source='{source}', target='{target}': "
        f"is_superperiod('{source}', '{target}') = {result_super}, "
        f"is_subperiod('{target}', '{source}') = {result_sub}"
    )

if __name__ == "__main__":
    # Run the property-based test
    test_subperiod_superperiod_symmetry_strings()
```

<details>

<summary>
**Failing input**: `source='Y', target='Y'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 28, in <module>
    test_subperiod_superperiod_symmetry_strings()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 16, in test_subperiod_superperiod_symmetry_strings
    def test_subperiod_superperiod_symmetry_strings(source, target):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 20, in test_subperiod_superperiod_symmetry_strings
    assert result_super == result_sub, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Symmetry violated for source='Y', target='Y': is_superperiod('Y', 'Y') = True, is_subperiod('Y', 'Y') = False
Falsifying example: test_subperiod_superperiod_symmetry_strings(
    source='Y',
    target='Y',
)
```
</details>

## Reproducing the Bug

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod

# Test with 'Y-JAN' frequency
freq = 'Y-JAN'
print(f"Testing frequency: '{freq}'")
print(f"is_superperiod('{freq}', '{freq}') = {is_superperiod(freq, freq)}")
print(f"is_subperiod('{freq}', '{freq}') = {is_subperiod(freq, freq)}")

# Check if symmetry property holds
super_result = is_superperiod(freq, freq)
sub_result = is_subperiod(freq, freq)

if super_result == sub_result:
    print(f"\n✓ Symmetry property holds: both return {super_result}")
else:
    print(f"\n✗ Symmetry violated!")
    print(f"  Expected: is_superperiod(a, b) == is_subperiod(b, a)")
    print(f"  Got: is_superperiod('{freq}', '{freq}') = {super_result}")
    print(f"       is_subperiod('{freq}', '{freq}') = {sub_result}")

# Test other annual frequencies
print("\n" + "="*50)
print("Testing other annual frequencies:")
annual_freqs = ['Y', 'Y-JAN', 'Y-FEB', 'Y-MAR', 'Y-DEC']

for freq in annual_freqs:
    super_res = is_superperiod(freq, freq)
    sub_res = is_subperiod(freq, freq)
    match = "✓" if super_res == sub_res else "✗"
    print(f"{match} '{freq}': is_superperiod={super_res}, is_subperiod={sub_res}")

# Demonstrate the expected behavior (symmetry property)
print("\n" + "="*50)
print("Demonstrating symmetry violation between different pairs:")
test_pairs = [('Y-JAN', 'Y-FEB'), ('Y', 'Y-JAN'), ('Y-MAR', 'Y')]

for source, target in test_pairs:
    super_res = is_superperiod(source, target)
    sub_res = is_subperiod(target, source)
    match = "✓" if super_res == sub_res else "✗"
    print(f"{match} is_superperiod('{source}', '{target}') = {super_res}")
    print(f"    is_subperiod('{target}', '{source}') = {sub_res}")
    if super_res != sub_res:
        print(f"    ^ Symmetry violated!")
    print()
```

<details>

<summary>
Output demonstrating the symmetry violation
</summary>
```
Testing frequency: 'Y-JAN'
is_superperiod('Y-JAN', 'Y-JAN') = True
is_subperiod('Y-JAN', 'Y-JAN') = False

✗ Symmetry violated!
  Expected: is_superperiod(a, b) == is_subperiod(b, a)
  Got: is_superperiod('Y-JAN', 'Y-JAN') = True
       is_subperiod('Y-JAN', 'Y-JAN') = False

==================================================
Testing other annual frequencies:
✗ 'Y': is_superperiod=True, is_subperiod=False
✗ 'Y-JAN': is_superperiod=True, is_subperiod=False
✗ 'Y-FEB': is_superperiod=True, is_subperiod=False
✗ 'Y-MAR': is_superperiod=True, is_subperiod=False
✗ 'Y-DEC': is_superperiod=True, is_subperiod=False

==================================================
Demonstrating symmetry violation between different pairs:
✓ is_superperiod('Y-JAN', 'Y-FEB') = False
    is_subperiod('Y-FEB', 'Y-JAN') = False

✓ is_superperiod('Y', 'Y-JAN') = False
    is_subperiod('Y-JAN', 'Y') = False

✓ is_superperiod('Y-MAR', 'Y') = False
    is_subperiod('Y', 'Y-MAR') = False
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Mathematical inconsistency**: By definition, if frequency A is a superperiod of frequency B, then B must be a subperiod of A. This is a fundamental property of period hierarchies that users and the pandas test suite expect.

2. **Test suite expectations**: The pandas test suite includes a test called `test_super_sub_symmetry` in `pandas/tests/tseries/frequencies/test_frequencies.py` that explicitly tests for this symmetry property, confirming it's an intended behavior.

3. **Reflexive property violation**: A frequency should be both a subperiod and superperiod of itself (reflexive property), but annual frequencies violate this - they are considered superperiods of themselves but not subperiods.

4. **Inconsistent implementation**: The `is_superperiod` function correctly handles the case when both frequencies are annual (lines 510-512 in frequencies.py), checking if they have the same month anchor. However, `is_subperiod` lacks this check entirely (lines 455-460).

## Relevant Context

The bug only affects annual frequencies (both plain 'Y' and month-anchored variants like 'Y-JAN', 'Y-FEB', etc.). All other frequency types correctly maintain the symmetry property. This suggests the issue is isolated to the annual frequency handling logic in `is_subperiod`.

The pandas source code location: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/tseries/frequencies.py`

Key functions:
- `is_subperiod`: Lines 438-486
- `is_superperiod`: Lines 489-519
- Helper functions: `_is_annual`, `get_rule_month`, `_quarter_months_conform`

## Proposed Fix

The fix adds the missing logic to check when both source and target are annual frequencies, mirroring the implementation in `is_superperiod`:

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -453,6 +453,10 @@ def is_subperiod(source, target) -> bool:
     target = _maybe_coerce_freq(target)

     if _is_annual(target):
+        if _is_annual(source):
+            # Both are annual - they're subperiods if they have the same month anchor
+            return get_rule_month(source) == get_rule_month(target)
+
         if _is_quarterly(source):
             return _quarter_months_conform(
                 get_rule_month(source), get_rule_month(target)
```