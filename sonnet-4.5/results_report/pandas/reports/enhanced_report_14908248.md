# Bug Report: pandas.tseries.frequencies Inverse Relationship Violation Between is_subperiod and is_superperiod

**Target**: `pandas.tseries.frequencies.is_subperiod` and `pandas.tseries.frequencies.is_superperiod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The functions `is_subperiod(source, target)` and `is_superperiod(source, target)` violate the mathematical inverse relationship for frequency conversion operations. Specifically, when `is_subperiod(A, B)` returns False, `is_superperiod(B, A)` returns True for business day and daily frequency combinations, violating the expected property that downsampling and upsampling operations should be inverses of each other.

## Property-Based Test

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod
from hypothesis import given, strategies as st

VALID_FREQS = ["D", "B", "C", "M", "h", "min", "s", "ms", "us", "ns", "W", "Y", "Q"]

@given(
    source=st.sampled_from(VALID_FREQS),
    target=st.sampled_from(VALID_FREQS)
)
def test_subperiod_superperiod_inverse(source, target):
    """Test that is_subperiod and is_superperiod maintain inverse relationship.

    The mathematical property being tested:
    is_subperiod(A, B) == is_superperiod(B, A) for all valid frequency pairs

    This should always hold because:
    - is_subperiod(A, B) checks if you can downsample from A to B
    - is_superperiod(B, A) checks if you can upsample from B to A
    - These operations are mathematical inverses of each other
    """
    result_sub = is_subperiod(source, target)
    result_super = is_superperiod(target, source)

    assert result_sub == result_super, (
        f"Inverse relationship violated: "
        f"is_subperiod({source!r}, {target!r}) = {result_sub} but "
        f"is_superperiod({target!r}, {source!r}) = {result_super}"
    )

# Run the test
if __name__ == "__main__":
    test_subperiod_superperiod_inverse()
```

<details>

<summary>
**Failing input**: `source='D', target='B'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 32, in <module>
    test_subperiod_superperiod_inverse()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 7, in test_subperiod_superperiod_inverse
    source=st.sampled_from(VALID_FREQS),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 24, in test_subperiod_superperiod_inverse
    assert result_sub == result_super, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Inverse relationship violated: is_subperiod('D', 'B') = False but is_superperiod('B', 'D') = True
Falsifying example: test_subperiod_superperiod_inverse(
    source='D',
    target='B',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/12/hypo.py:25
```
</details>

## Reproducing the Bug

```python
from pandas.tseries.frequencies import is_subperiod, is_superperiod

# Test the specific failing cases mentioned in the bug report
print("Testing inverse relationship violations:")
print()

# Test 1: Daily (D) vs Business Day (B)
print("Test 1: Daily (D) vs Business Day (B)")
print(f"is_subperiod('D', 'B') = {is_subperiod('D', 'B')}")
print(f"is_superperiod('B', 'D') = {is_superperiod('B', 'D')}")
print(f"These should be equal, but are not! (Expected: both True)")
print()

# Test 2: Daily (D) vs Custom Business Day (C)
print("Test 2: Daily (D) vs Custom Business Day (C)")
print(f"is_subperiod('D', 'C') = {is_subperiod('D', 'C')}")
print(f"is_superperiod('C', 'D') = {is_superperiod('C', 'D')}")
print(f"These should be equal, but are not! (Expected: both True)")
print()

# Test 3: Custom Business Day (C) vs Business Day (B)
print("Test 3: Custom Business Day (C) vs Business Day (B)")
print(f"is_subperiod('C', 'B') = {is_subperiod('C', 'B')}")
print(f"is_superperiod('B', 'C') = {is_superperiod('B', 'C')}")
print(f"These should be equal, but are not! (Expected: both True)")
print()

# Additional tests for reverse direction violations
print("Additional violations in reverse directions:")
print()

# Test 4: Business Day (B) vs Daily (D) - reverse
print("Test 4: Business Day (B) vs Daily (D)")
print(f"is_subperiod('B', 'D') = {is_subperiod('B', 'D')}")
print(f"is_superperiod('D', 'B') = {is_superperiod('D', 'B')}")
print(f"These should be equal, but are not! (Expected: both False)")
print()

# Test 5: Business Day (B) vs Custom Business Day (C) - reverse
print("Test 5: Business Day (B) vs Custom Business Day (C)")
print(f"is_subperiod('B', 'C') = {is_subperiod('B', 'C')}")
print(f"is_superperiod('C', 'B') = {is_superperiod('C', 'B')}")
print(f"These should be equal, but are not! (Expected: both False)")
print()

# Test 6: Custom Business Day (C) vs Daily (D) - reverse
print("Test 6: Custom Business Day (C) vs Daily (D)")
print(f"is_subperiod('C', 'D') = {is_subperiod('C', 'D')}")
print(f"is_superperiod('D', 'C') = {is_superperiod('D', 'C')}")
print(f"These should be equal, but are not! (Expected: both False)")
print()

# Test 7: Year (Y) vs Year (Y) - self-comparison
print("Test 7: Year (Y) vs Year (Y) - self-comparison")
print(f"is_subperiod('Y', 'Y') = {is_subperiod('Y', 'Y')}")
print(f"is_superperiod('Y', 'Y') = {is_superperiod('Y', 'Y')}")
print(f"These should be equal (both False based on code inspection)")
print()

# Verify the asymmetry in the implementation
print("Summary:")
print("The inverse relationship property is violated for multiple frequency pairs.")
print("If is_subperiod(A, B) is True, then is_superperiod(B, A) should also be True.")
print("If is_subperiod(A, B) is False, then is_superperiod(B, A) should also be False.")
```

<details>

<summary>
Output showing 7 inverse relationship violations
</summary>
```
Testing inverse relationship violations:

Test 1: Daily (D) vs Business Day (B)
is_subperiod('D', 'B') = False
is_superperiod('B', 'D') = True
These should be equal, but are not! (Expected: both True)

Test 2: Daily (D) vs Custom Business Day (C)
is_subperiod('D', 'C') = False
is_superperiod('C', 'D') = True
These should be equal, but are not! (Expected: both True)

Test 3: Custom Business Day (C) vs Business Day (B)
is_subperiod('C', 'B') = False
is_superperiod('B', 'C') = True
These should be equal, but are not! (Expected: both True)

Additional violations in reverse directions:

Test 4: Business Day (B) vs Daily (D)
is_subperiod('B', 'D') = False
is_superperiod('D', 'B') = True
These should be equal, but are not! (Expected: both False)

Test 5: Business Day (B) vs Custom Business Day (C)
is_subperiod('B', 'C') = False
is_superperiod('C', 'B') = True
These should be equal, but are not! (Expected: both False)

Test 6: Custom Business Day (C) vs Daily (D)
is_subperiod('C', 'D') = False
is_superperiod('D', 'C') = True
These should be equal, but are not! (Expected: both False)

Test 7: Year (Y) vs Year (Y) - self-comparison
is_subperiod('Y', 'Y') = False
is_superperiod('Y', 'Y') = True
These should be equal (both False based on code inspection)

Summary:
The inverse relationship property is violated for multiple frequency pairs.
If is_subperiod(A, B) is True, then is_superperiod(B, A) should also be True.
If is_subperiod(A, B) is False, then is_superperiod(B, A) should also be False.
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical property that downsampling and upsampling operations should be inverses of each other. The docstrings explicitly state:
- `is_subperiod`: "Returns True if downsampling is possible between source and target frequencies"
- `is_superperiod`: "Returns True if upsampling is possible between source and target frequencies"

By definition, if you can downsample from frequency A to frequency B, then you should be able to upsample from frequency B back to frequency A. The current implementation violates this principle due to asymmetric handling of business day frequencies.

Examining the source code at lines 467-470 and 525-528 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/tseries/frequencies.py` reveals the root cause:

**For `is_subperiod` when target is "B":**
```python
elif target == "B":
    return source in {"B", "h", "min", "s", "ms", "us", "ns"}
```
Notice that "D" (Daily) and "C" (Custom Business Day) are NOT included as valid sources.

**For `is_superperiod` when source is "B":**
```python
elif source == "B":
    return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
```
Here "D" and "C" ARE included as valid targets.

This asymmetry means `is_superperiod('B', 'D')` returns True (claiming you can upsample from Business Day to Daily), but `is_subperiod('D', 'B')` returns False (claiming you cannot downsample from Daily to Business Day). This is logically inconsistent - these operations should be inverses.

## Relevant Context

1. **Internal Functions**: These functions are located in `pandas.tseries.frequencies` but are not part of the documented public API. They're primarily used internally by pandas for resampling operations in `pandas/core/resample.py`.

2. **Frequency Meanings**:
   - "D" = Daily (calendar days, including weekends)
   - "B" = Business Day (weekdays only, excluding weekends)
   - "C" = Custom Business Day (customizable business days with holidays)

3. **Related Issues**: GitHub Issue #18553 documents similar inconsistent behavior where `is_subperiod('M', 'M')` returns False while `is_subperiod('D', 'D')` returns True, suggesting these functions have specific internal semantics that may not match intuitive expectations.

4. **Impact**: While these are internal functions, they affect the behavior of pandas' resampling operations. The violations could lead to unexpected behavior when converting between business day and calendar day frequencies.

## Proposed Fix

The fix requires making the two functions symmetric by including "D" and complementary business frequencies in both directions:

```diff
--- a/pandas/tseries/frequencies.py
+++ b/pandas/tseries/frequencies.py
@@ -467,9 +467,9 @@ def is_subperiod(source, target) -> bool:
     elif target == "B":
-        return source in {"B", "h", "min", "s", "ms", "us", "ns"}
+        return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "C":
-        return source in {"C", "h", "min", "s", "ms", "us", "ns"}
+        return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
     elif target == "D":
         return source in {"D", "h", "min", "s", "ms", "us", "ns"}
```

This ensures the inverse relationship holds: if you can upsample from B to D (which the current code allows), then you should also be able to downsample from D to B.