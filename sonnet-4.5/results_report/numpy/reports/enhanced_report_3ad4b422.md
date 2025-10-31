# Bug Report: numpy.busday_count Antisymmetry Property Violation

**Target**: `numpy.busday_count`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.busday_count` function violates the antisymmetry property when counting business days between dates where one is a non-business day and the other is a business day, resulting in `busday_count(a, b) + busday_count(b, a) ≠ 0`.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st


datetime_strategy = st.integers(min_value=0, max_value=20000).map(
    lambda days: np.datetime64('2000-01-01') + np.timedelta64(days, 'D')
)


@given(datetime_strategy, datetime_strategy)
@settings(max_examples=1000)
def test_busday_count_antisymmetric(date1, date2):
    count_forward = np.busday_count(date1, date2)
    count_backward = np.busday_count(date2, date1)
    assert count_forward == -count_backward, f"Antisymmetry violated: busday_count({date1}, {date2})={count_forward}, busday_count({date2}, {date1})={count_backward}"


if __name__ == "__main__":
    test_busday_count_antisymmetric()
```

<details>

<summary>
**Failing input**: `date1=np.datetime64('2000-01-01'), date2=np.datetime64('2000-01-03')`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 19, in <module>
    test_busday_count_antisymmetric()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 11, in test_busday_count_antisymmetric
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 15, in test_busday_count_antisymmetric
    assert count_forward == -count_backward, f"Antisymmetry violated: busday_count({date1}, {date2})={count_forward}, busday_count({date2}, {date1})={count_backward}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Antisymmetry violated: busday_count(2000-01-01, 2000-01-03)=0, busday_count(2000-01-03, 2000-01-01)=-1
Falsifying example: test_busday_count_antisymmetric(
    date1=(lambda days: np.datetime64('2000-01-01') + np.timedelta64(days, 'D'))(
        0,
    ),
    date2=(lambda days: np.datetime64('2000-01-01') + np.timedelta64(days, 'D'))(
        2,
    ),
)
```
</details>

## Reproducing the Bug

```python
import numpy as np

# Testing the antisymmetry property violation
saturday = np.datetime64('2000-01-01')  # Saturday (non-business day)
monday = np.datetime64('2000-01-03')    # Monday (business day)

# Count business days forward (Saturday to Monday)
count_forward = np.busday_count(saturday, monday)

# Count business days backward (Monday to Saturday)
count_backward = np.busday_count(monday, saturday)

print("Testing busday_count antisymmetry property:")
print("=" * 50)
print(f"Date 1 (Saturday): {saturday}")
print(f"Date 2 (Monday): {monday}")
print()
print(f"busday_count(Saturday, Monday) = {count_forward}")
print(f"busday_count(Monday, Saturday) = {count_backward}")
print()
print("Antisymmetry property check:")
print(f"Expected: count_forward = -count_backward")
print(f"Expected: {count_forward} = {-count_backward}")
print(f"Actual result: {count_forward} {'==' if count_forward == -count_backward else '!='} {-count_backward}")
print()
print(f"Property violated: {count_forward != -count_backward}")

# Let's also test with Tuesday to understand the pattern
tuesday = np.datetime64('2000-01-04')
count_sat_tue = np.busday_count(saturday, tuesday)
count_tue_sat = np.busday_count(tuesday, saturday)

print()
print("Additional test with Tuesday:")
print(f"busday_count(Saturday, Tuesday) = {count_sat_tue}")
print(f"busday_count(Tuesday, Saturday) = {count_tue_sat}")
print(f"Expected: {count_sat_tue} = {-count_tue_sat}")
print(f"Property violated: {count_sat_tue != -count_tue_sat}")
```

<details>

<summary>
Antisymmetry property violation demonstrated
</summary>
```
Testing busday_count antisymmetry property:
==================================================
Date 1 (Saturday): 2000-01-01
Date 2 (Monday): 2000-01-03

busday_count(Saturday, Monday) = 0
busday_count(Monday, Saturday) = -1

Antisymmetry property check:
Expected: count_forward = -count_backward
Expected: 0 = 1
Actual result: 0 != 1

Property violated: True

Additional test with Tuesday:
busday_count(Saturday, Tuesday) = 1
busday_count(Tuesday, Saturday) = -2
Expected: 1 = 2
Property violated: True
```
</details>

## Why This Is A Bug

The numpy documentation for `busday_count` states that it "counts the number of valid days between `begindates` and `enddates`, not including the day of `enddates`". This describes a half-open interval [begin, end). Any interval counting function should satisfy the antisymmetry property: `count(a, b) = -count(b, a)`, meaning that reversing the direction of counting should negate the result.

However, the function uses inconsistent interval semantics depending on the direction:
- **Forward counting** (begin < end): Uses [begin, end) interval - includes begin if it's a business day, excludes end
- **Backward counting** (begin > end): Appears to use (end, begin] interval - excludes end, but includes begin in the count

This inconsistency manifests when:
1. Saturday to Monday: Returns 0 (counts days in [Sat, Mon) = {Sat, Sun}, neither are business days)
2. Monday to Saturday: Returns -1 (appears to count Monday as a business day when going backward)

The violation becomes clearer with Tuesday:
- Saturday to Tuesday: Returns 1 (counts Monday in [Sat, Tue))
- Tuesday to Saturday: Returns -2 (counts both Monday and Tuesday)

This breaks the mathematical property that `busday_count(a, b) + busday_count(b, a) = 0`, which users would reasonably expect for any counting function. This could lead to incorrect business day calculations in financial applications, scheduling systems, and any code that relies on symmetric date range calculations.

## Relevant Context

The bug appears to stem from how the function handles the boundary conditions when the direction is reversed. The documentation at https://numpy.org/doc/stable/reference/generated/numpy.busday_count.html specifies that the function uses a half-open interval excluding the end date, but the implementation seems to apply different logic for backward counting.

This issue affects numpy version 2.3.0 (current as of testing) and likely earlier versions. The weekmask defaults to '1111100' (Monday through Friday as business days), making Saturday and Sunday non-business days.

## Proposed Fix

The fix requires modifying the C implementation to ensure consistent interval semantics regardless of counting direction. When `begindate > enddate`, the function should:

1. Swap the dates to get the forward interval
2. Count business days in the standard [begin, end) interval
3. Negate the result to account for the reversed direction

This would ensure that `busday_count(a, b)` always uses [a, b) interval semantics and maintains the antisymmetry property. The specific fix would need to be applied to the underlying C code in numpy's datetime module, likely in the `_count_busdaycal` function or its callers.