# Bug Report: pandas.tseries.frequencies is_superperiod/is_subperiod Antisymmetry and Inverse Relationship Violation

**Target**: `pandas.tseries.frequencies.is_superperiod` and `pandas.tseries.frequencies.is_subperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_superperiod` and `is_subperiod` functions violate both the antisymmetric property and inverse relationship: both `is_superperiod('D', 'B')` and `is_superperiod('B', 'D')` return `True`, and neither `is_subperiod` call returns the expected inverse result.

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


if __name__ == "__main__":
    test_is_superperiod_subperiod_inverse()
```

<details>

<summary>
**Failing input**: `source='D'`, `target='B'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 20, in <module>
    test_is_superperiod_subperiod_inverse()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 12, in test_is_superperiod_subperiod_inverse
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 15, in test_is_superperiod_subperiod_inverse
    assert freq.is_subperiod(target, source), \
           ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
AssertionError: is_superperiod('D', 'B') is True but is_subperiod('B', 'D') is False
Falsifying example: test_is_superperiod_subperiod_inverse(
    source='D',
    target='B',
)
```
</details>

## Reproducing the Bug

```python
import pandas.tseries.frequencies as freq

# Testing the antisymmetric property violation
result1 = freq.is_superperiod('D', 'B')
result2 = freq.is_superperiod('B', 'D')

print(f"is_superperiod('D', 'B') = {result1}")
print(f"is_superperiod('B', 'D') = {result2}")

if result1 and result2:
    print("BUG CONFIRMED: Both return True (violates antisymmetry)")
else:
    print("Antisymmetry property is satisfied")

print("\n--- Testing inverse relationship ---")
# Testing the inverse relationship between is_superperiod and is_subperiod
result3 = freq.is_subperiod('B', 'D')
result4 = freq.is_subperiod('D', 'B')

print(f"is_subperiod('B', 'D') = {result3}")
print(f"is_subperiod('D', 'B') = {result4}")

print("\n--- Expected behavior ---")
print("If is_superperiod('D', 'B') is True, then:")
print("  - is_superperiod('B', 'D') should be False (antisymmetry)")
print("  - is_subperiod('B', 'D') should be True (inverse relationship)")
```

<details>

<summary>
Output demonstrating the bug
</summary>
```
is_superperiod('D', 'B') = True
is_superperiod('B', 'D') = True
BUG CONFIRMED: Both return True (violates antisymmetry)

--- Testing inverse relationship ---
is_subperiod('B', 'D') = False
is_subperiod('D', 'B') = False

--- Expected behavior ---
If is_superperiod('D', 'B') is True, then:
  - is_superperiod('B', 'D') should be False (antisymmetry)
  - is_subperiod('B', 'D') should be True (inverse relationship)
```
</details>

## Why This Is A Bug

The `is_superperiod` and `is_subperiod` functions violate two fundamental mathematical properties that are essential for any hierarchical relationship:

1. **Antisymmetry Violation**: For any frequencies A and B where A ≠ B, if `is_superperiod(A, B)` returns `True`, then `is_superperiod(B, A)` must return `False`. Currently, both `is_superperiod('D', 'B')` and `is_superperiod('B', 'D')` return `True`, which is logically impossible - two different frequencies cannot both be superperiods of each other.

2. **Inverse Relationship Violation**: The functions `is_superperiod` and `is_subperiod` should be inverse operations. If `is_superperiod(A, B)` returns `True`, then `is_subperiod(B, A)` should also return `True`. Currently, when `is_superperiod('D', 'B')` returns `True`, `is_subperiod('B', 'D')` returns `False`, breaking this expected inverse relationship.

The docstrings for these functions state they determine if "upsampling" (for `is_superperiod`) or "downsampling" (for `is_subperiod`) is possible between frequencies. The current behavior makes it impossible to reliably determine the direction of frequency conversion.

## Relevant Context

The bug occurs specifically with 'D' (Day) and 'B' (BusinessDay) frequencies, as well as 'C' (CustomBusinessDay). These represent different calendar systems:
- 'D' represents all calendar days (including weekends)
- 'B' represents business days only (excluding weekends)
- 'C' represents custom business days

These functions are exported in the module's `__all__` list (line 595-602 of frequencies.py), indicating they are part of the public API and intended for external use. The functions are used internally by pandas for frequency conversion and resampling operations, so incorrect behavior could affect data resampling accuracy.

Code location: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/tseries/frequencies.py`
- `is_superperiod` function: lines 489-544
- `is_subperiod` function: lines 434-486
- Problematic lines: 526, 528, 530 (in `is_superperiod`)

## Proposed Fix

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -523,11 +523,11 @@ def is_superperiod(source, target) -> bool:
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
     elif source == "h":
         return target in {"h", "min", "s", "ms", "us", "ns"}
     elif source == "min":
```