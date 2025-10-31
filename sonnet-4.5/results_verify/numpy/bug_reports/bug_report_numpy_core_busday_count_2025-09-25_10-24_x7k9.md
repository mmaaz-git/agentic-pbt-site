# Bug Report: numpy.core.busday_count Antisymmetry Violation

**Target**: `numpy.core.busday_count`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.core.busday_count` violates the antisymmetry property when the begin date is a non-business day and the end date is a business day. The function uses inconsistent interval semantics for forward vs backward counting, causing `busday_count(a, b) + busday_count(b, a) ≠ 0`.

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
    assert count_forward == -count_backward
```

**Failing input**: `date1=np.datetime64('2000-01-01')`, `date2=np.datetime64('2000-01-03')`

## Reproducing the Bug

```python
import numpy as np

saturday = np.datetime64('2000-01-01')
monday = np.datetime64('2000-01-03')

count_forward = np.busday_count(saturday, monday)
count_backward = np.busday_count(monday, saturday)

print(f"busday_count(Saturday, Monday) = {count_forward}")
print(f"busday_count(Monday, Saturday) = {count_backward}")
print(f"\nExpected (antisymmetry): {count_forward} = -{count_backward}")
print(f"Actual: {count_forward} != {-count_backward}")
```

Output:
```
busday_count(Saturday, Monday) = 0
busday_count(Monday, Saturday) = -1

Expected (antisymmetry): 0 = --1
Actual: 0 != 1
```

## Why This Is A Bug

The documentation states that `busday_count` counts "the number of valid days between begindates and enddates, not including the day of enddates", defining a half-open interval [begin, end). Any interval counting function should satisfy antisymmetry: `count(a, b) = -count(b, a)`.

The function exhibits inconsistent interval semantics:
- **Forward** (begin < end): Uses [begin, end) - includes begin, excludes end
- **Backward** (begin > end): Uses (end, begin] - excludes end, includes begin

Evidence:
- `busday_count(Sat, Mon)` = 0 (interval [Sat, Mon) = {Sat, Sun}, 0 business days) ✓
- `busday_count(Mon, Sat)` = -1 (interval (Sat, Mon] = {Sun, Mon}, 1 business day Mon)
- `busday_count(Sat, Tue)` = 1 (interval [Sat, Tue) = {Sat, Sun, Mon}, 1 business day) ✓
- `busday_count(Tue, Sat)` = -2 (interval (Sat, Tue] = {Sun, Mon, Tue}, 2 business days)

This violates antisymmetry and produces incorrect results when users expect symmetric behavior, particularly affecting date range calculations in business day calendars.

## Fix

The backward counting should use the same half-open interval [begin, end) as forward counting. When `begindate > enddate`, the function should:
1. Count business days in [begindate, enddate) using the same interval semantics
2. Negate the result

This ensures `busday_count(a, b) + busday_count(b, a) = 0` for all dates a, b.

The fix requires modifying the C implementation to apply consistent interval semantics regardless of direction.