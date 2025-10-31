# Bug Report: pandas.tseries.frequencies Circular Frequency Relationship Logic Error

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod()` and `is_superperiod()` violate their mathematical inverse relationship and create logically impossible circular dependencies for business day ('B') and calendar day ('D') frequencies.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.tseries import frequencies

freq_strings = st.sampled_from([
    'D', 'h', 'min', 's', 'ms', 'us', 'ns',
    'W', 'ME', 'MS', 'QE', 'QS', 'YE', 'YS',
    'B', 'BME', 'BMS', 'BQE', 'BQS', 'BYE', 'BYS'
])

@given(freq_strings, freq_strings)
@settings(max_examples=500)
def test_subperiod_superperiod_inverse_relationship(source, target):
    is_sub = frequencies.is_subperiod(source, target)
    is_super = frequencies.is_superperiod(target, source)
    assert is_sub == is_super, f"Failed for source={source}, target={target}: is_subperiod={is_sub}, is_superperiod={is_super}"

# Run the test
test_subperiod_superperiod_inverse_relationship()
```

<details>

<summary>
**Failing input**: `source='D', target='B'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 18, in <module>
    test_subperiod_superperiod_inverse_relationship()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 11, in test_subperiod_superperiod_inverse_relationship
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 15, in test_subperiod_superperiod_inverse_relationship
    assert is_sub == is_super, f"Failed for source={source}, target={target}: is_subperiod={is_sub}, is_superperiod={is_super}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Failed for source=D, target=B: is_subperiod=False, is_superperiod=True
Falsifying example: test_subperiod_superperiod_inverse_relationship(
    source='D',
    target='B',
)
```
</details>

## Reproducing the Bug

```python
from pandas.tseries import frequencies

print("Testing inverse relationship between is_subperiod and is_superperiod:")
print("=" * 60)

# Test 1: Check the inverse relationship
print("\nTest 1: Inverse Relationship (should be equal):")
is_sub_d_b = frequencies.is_subperiod('D', 'B')
is_super_b_d = frequencies.is_superperiod('B', 'D')
print(f"is_subperiod('D', 'B') = {is_sub_d_b}")
print(f"is_superperiod('B', 'D') = {is_super_b_d}")
print(f"Are they equal? {is_sub_d_b == is_super_b_d}")

# This assertion should pass but doesn't
try:
    assert is_sub_d_b == is_super_b_d
    print("✓ Inverse relationship holds")
except AssertionError:
    print("✗ FAILED: Inverse relationship violated!")

print("\n" + "=" * 60)

# Test 2: Check both superperiods (logically impossible)
print("\nTest 2: Both Superperiods (at most one should be True):")
is_super_d_b = frequencies.is_superperiod('D', 'B')
is_super_b_d_2 = frequencies.is_superperiod('B', 'D')
print(f"is_superperiod('D', 'B') = {is_super_d_b}")
print(f"is_superperiod('B', 'D') = {is_super_b_d_2}")

# Both cannot be True (circular relationship)
try:
    assert not (is_super_d_b and is_super_b_d_2)
    print("✓ No circular superperiod relationship")
except AssertionError:
    print("✗ FAILED: Circular superperiod relationship exists!")

print("\n" + "=" * 60)

# Additional context
print("\nAdditional Context:")
print(f"is_subperiod('B', 'D') = {frequencies.is_subperiod('B', 'D')}")
print(f"is_subperiod('D', 'B') = {frequencies.is_subperiod('D', 'B')}")

print("\nSummary:")
print("- 'D' (calendar day) represents all 7 days per week")
print("- 'B' (business day) represents only 5 days per week (Mon-Fri)")
print("- Therefore, 'D' is more frequent than 'B'")
print("- Expected: is_subperiod('D', 'B') should be True (can downsample)")
print("- Expected: is_superperiod('D', 'B') should be False")
```

<details>

<summary>
Output showing two distinct logic violations
</summary>
```
Testing inverse relationship between is_subperiod and is_superperiod:
============================================================

Test 1: Inverse Relationship (should be equal):
is_subperiod('D', 'B') = False
is_superperiod('B', 'D') = True
Are they equal? False
✗ FAILED: Inverse relationship violated!

============================================================

Test 2: Both Superperiods (at most one should be True):
is_superperiod('D', 'B') = True
is_superperiod('B', 'D') = True
✗ FAILED: Circular superperiod relationship exists!

============================================================

Additional Context:
is_subperiod('B', 'D') = False
is_subperiod('D', 'B') = False

Summary:
- 'D' (calendar day) represents all 7 days per week
- 'B' (business day) represents only 5 days per week (Mon-Fri)
- Therefore, 'D' is more frequent than 'B'
- Expected: is_subperiod('D', 'B') should be True (can downsample)
- Expected: is_superperiod('D', 'B') should be False
```
</details>

## Why This Is A Bug

This bug violates two fundamental mathematical properties:

1. **Inverse Relationship Violation**: According to the docstrings, `is_subperiod` checks if downsampling is possible from source to target, while `is_superperiod` checks if upsampling is possible. These are inverse operations - if you can downsample from A to B, then you can upsample from B to A. However:
   - `is_subperiod('D', 'B')` returns `False`
   - `is_superperiod('B', 'D')` returns `True`
   - These should be equal, but they're not

2. **Circular Dependency**: Two frequencies cannot simultaneously be superperiods of each other, as this would mean each is both more and less frequent than the other. However:
   - `is_superperiod('D', 'B')` returns `True`
   - `is_superperiod('B', 'D')` returns `True`
   - Both cannot be True - this is logically impossible

The root cause is that calendar days ('D') occur 7 times per week while business days ('B') occur only 5 times per week, making 'D' more frequent than 'B'. The functions should reflect this hierarchy.

## Relevant Context

The bug exists in the implementation at `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/tseries/frequencies.py`:

- Lines 467-468 in `is_subperiod`: For target='B', only allows source in `{"B", "h", "min", "s", "ms", "us", "ns"}` - missing 'D' and 'C'
- Lines 525-530 in `is_superperiod`: For source='B', 'C', and 'D', they all include each other in their valid target sets: `{"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}`

This affects real-world time series resampling operations where users need to determine valid frequency conversions between business and calendar days.

Documentation:
- `is_subperiod` docstring (line 436): "Returns True if downsampling is possible between source and target frequencies"
- `is_superperiod` docstring (line 491): "Returns True if upsampling is possible between source and target frequencies"

## Proposed Fix

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -465,7 +465,7 @@ def is_subperiod(source, target) -> bool:
     elif _is_weekly(target):
         return source in {target, "D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "B":
-        return source in {"B", "h", "min", "s", "ms", "us", "ns"}
+        return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "C":
         return source in {"C", "h", "min", "s", "ms", "us", "ns"}
     elif target == "D":
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