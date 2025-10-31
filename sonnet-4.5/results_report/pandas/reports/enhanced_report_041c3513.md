# Bug Report: pandas.tseries.frequencies is_subperiod and is_superperiod Are Not Inverse Operations

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod` and `is_superperiod` violate their documented contract of being inverse operations. When `is_superperiod(source, target)` returns `True`, `is_subperiod(target, source)` should also return `True`, but this fails for all combinations of day-level frequencies ('D', 'B', 'C').

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test showing that is_subperiod and is_superperiod are not inverse operations
"""

from hypothesis import given, settings, strategies as st
from pandas.tseries.frequencies import is_subperiod, is_superperiod

freq_strings = st.sampled_from([
    "D", "B", "C", "h", "min", "s", "ms", "us", "ns",
    "M", "BM", "W", "Y", "Q",
])

@settings(max_examples=1000)
@given(source=freq_strings, target=freq_strings)
def test_superperiod_subperiod_inverse(source, target):
    if is_superperiod(source, target):
        assert is_subperiod(target, source), (
            f"If {source} is_superperiod of {target}, "
            f"then {target} should be is_subperiod of {source}"
        )

if __name__ == "__main__":
    # Run the test to find failing cases
    test_superperiod_subperiod_inverse()
```

<details>

<summary>
**Failing input**: `source='D', target='B'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 25, in <module>
    test_superperiod_subperiod_inverse()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 15, in test_superperiod_subperiod_inverse
    @given(source=freq_strings, target=freq_strings)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 18, in test_superperiod_subperiod_inverse
    assert is_subperiod(target, source), (
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^
AssertionError: If D is_superperiod of B, then B should be is_subperiod of D
Falsifying example: test_superperiod_subperiod_inverse(
    source='D',
    target='B',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/20/hypo.py:19
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the is_subperiod/is_superperiod inverse bug in pandas.tseries.frequencies
"""

from pandas.tseries.frequencies import is_subperiod, is_superperiod

# Test case 1: D (calendar day) and B (business day)
print("Test case 1: D and B")
print(f"is_superperiod('D', 'B') = {is_superperiod('D', 'B')}")
print(f"is_subperiod('B', 'D') = {is_subperiod('B', 'D')}")
print(f"  Expected: If superperiod returns True, subperiod should also return True")
print()

# Test case 2: D (calendar day) and C (custom business day)
print("Test case 2: D and C")
print(f"is_superperiod('D', 'C') = {is_superperiod('D', 'C')}")
print(f"is_subperiod('C', 'D') = {is_subperiod('C', 'D')}")
print(f"  Expected: If superperiod returns True, subperiod should also return True")
print()

# Test case 3: B (business day) and D (calendar day)
print("Test case 3: B and D")
print(f"is_superperiod('B', 'D') = {is_superperiod('B', 'D')}")
print(f"is_subperiod('D', 'B') = {is_subperiod('D', 'B')}")
print(f"  Expected: If superperiod returns True, subperiod should also return True")
print()

# Test case 4: B (business day) and C (custom business day)
print("Test case 4: B and C")
print(f"is_superperiod('B', 'C') = {is_superperiod('B', 'C')}")
print(f"is_subperiod('C', 'B') = {is_subperiod('C', 'B')}")
print(f"  Expected: If superperiod returns True, subperiod should also return True")
print()

# Test case 5: C (custom business day) and D (calendar day)
print("Test case 5: C and D")
print(f"is_superperiod('C', 'D') = {is_superperiod('C', 'D')}")
print(f"is_subperiod('D', 'C') = {is_subperiod('D', 'C')}")
print(f"  Expected: If superperiod returns True, subperiod should also return True")
print()

# Test case 6: C (custom business day) and B (business day)
print("Test case 6: C and B")
print(f"is_superperiod('C', 'B') = {is_superperiod('C', 'B')}")
print(f"is_subperiod('B', 'C') = {is_subperiod('B', 'C')}")
print(f"  Expected: If superperiod returns True, subperiod should also return True")
```

<details>

<summary>
All six test cases demonstrate the inverse property violation
</summary>
```
Test case 1: D and B
is_superperiod('D', 'B') = True
is_subperiod('B', 'D') = False
  Expected: If superperiod returns True, subperiod should also return True

Test case 2: D and C
is_superperiod('D', 'C') = True
is_subperiod('C', 'D') = False
  Expected: If superperiod returns True, subperiod should also return True

Test case 3: B and D
is_superperiod('B', 'D') = True
is_subperiod('D', 'B') = False
  Expected: If superperiod returns True, subperiod should also return True

Test case 4: B and C
is_superperiod('B', 'C') = True
is_subperiod('C', 'B') = False
  Expected: If superperiod returns True, subperiod should also return True

Test case 5: C and D
is_superperiod('C', 'D') = True
is_subperiod('D', 'C') = False
  Expected: If superperiod returns True, subperiod should also return True

Test case 6: C and B
is_superperiod('C', 'B') = True
is_subperiod('B', 'C') = False
  Expected: If superperiod returns True, subperiod should also return True
```
</details>

## Why This Is A Bug

The functions `is_subperiod` and `is_superperiod` have clear, complementary purposes according to their docstrings:

- `is_subperiod`: "Returns True if downsampling is possible between source and target frequencies"
- `is_superperiod`: "Returns True if upsampling is possible between source and target frequencies"

These operations are mathematical inverses: if you can downsample from frequency A to frequency B, then by definition you must be able to upsample from B back to A. The current implementation violates this fundamental property.

The root cause is an asymmetry in the implementation at lines 467-472 and 525-530 of `/pandas/tseries/frequencies.py`:

1. In `is_superperiod`, when the source is "D", "B", or "C", the function correctly returns True for any target in the set {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}. This means all three day-level frequencies are treated as equivalent super-periods.

2. However, in `is_subperiod`, when the target is "D", "B", or "C", each only accepts a restricted subset of sources that excludes the other day-level frequencies:
   - target="B" only accepts sources in {"B", "h", "min", "s", "ms", "us", "ns"} - missing "D" and "C"
   - target="C" only accepts sources in {"C", "h", "min", "s", "ms", "us", "ns"} - missing "D" and "B"
   - target="D" only accepts sources in {"D", "h", "min", "s", "ms", "us", "ns"} - missing "B" and "C"

This asymmetry breaks the inverse relationship between the two functions.

## Relevant Context

The day-level frequencies have the following meanings in pandas:
- 'D': Calendar day frequency
- 'B': Business day frequency (excludes weekends)
- 'C': Custom business day frequency (user-defined holidays)

These are commonly used in financial time series analysis where converting between different day conventions is a regular operation. The bug affects users who rely on these functions to validate frequency conversions before performing resampling operations.

Source code location: https://github.com/pandas-dev/pandas/blob/main/pandas/tseries/frequencies.py

## Proposed Fix

The fix requires making the `is_subperiod` function symmetric with `is_superperiod` for day-level frequencies:

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -465,11 +465,11 @@ def is_subperiod(source, target) -> bool:
     elif _is_weekly(target):
         return source in {target, "D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "B":
-        return source in {"B", "h", "min", "s", "ms", "us", "ns"}
+        return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "C":
-        return source in {"C", "h", "min", "s", "ms", "us", "ns"}
+        return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "D":
-        return source in {"D", "h", "min", "s", "ms", "us", "ns"}
+        return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "h":
         return source in {"h", "min", "s", "ms", "us", "ns"}
     elif target == "min":
```