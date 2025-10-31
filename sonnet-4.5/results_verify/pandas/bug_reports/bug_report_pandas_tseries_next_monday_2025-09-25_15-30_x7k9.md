# Bug Report: pandas.tseries.holiday.next_monday Misleading Name and Incomplete Documentation

**Target**: `pandas.tseries.holiday.next_monday`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The function `next_monday` has a misleading name that violates the principle of least surprise. Based on its name, users would expect it to return the next Monday from any given date, but it actually only shifts weekend dates (Saturday/Sunday) to Monday and returns weekday dates unchanged. Additionally, the docstring is incomplete and doesn't document the behavior for weekdays.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from datetime import datetime
from pandas.tseries.holiday import next_monday


@given(st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2030, 1, 1)))
@settings(max_examples=200)
def test_next_monday_name_contract(dt):
    result = next_monday(dt)

    if dt.weekday() in [1, 2, 3, 4]:
        assert result.weekday() == 0, \
            f"next_monday from {dt.strftime('%A')} should return next Monday, not {result.strftime('%A')}"
```

**Failing input**: `datetime.datetime(2000, 2, 1, 0, 0)` (Tuesday)

## Reproducing the Bug

```python
from pandas.tseries.holiday import next_monday
from datetime import datetime

thursday = datetime(2020, 6, 4)
result = next_monday(thursday)

print(f"Input: {thursday.strftime('%A')}")
print(f"Result: {result.strftime('%A')}")
print(f"Result weekday: {result.weekday()}")

assert result.weekday() == 0
```

Output:
```
Input: Thursday
Result: Thursday
Result weekday: 3
AssertionError
```

## Why This Is A Bug

This violates the contract established by the function name and causes confusion:

1. **Misleading Name**: The function name `next_monday` strongly implies it returns the next Monday from any given date, similar to functions like `next_workday`. However, it only shifts weekend dates to Monday and returns the same date for Monday-Friday.

2. **Incomplete Documentation**: The docstring states:
   ```
   If holiday falls on Saturday, use following Monday instead;
   if holiday falls on Sunday, use Monday instead
   ```
   This doesn't document what happens for weekdays (Monday-Friday), leaving users to guess.

3. **Inconsistent with Similar Functions**: Other functions in the same module have clearer names:
   - `sunday_to_monday`: clearly indicates Sunday → Monday transformation
   - `weekend_to_monday`: clearly indicates weekend → Monday transformation
   - `next_workday`: finds the next workday (as name suggests)

4. **Violates Principle of Least Surprise**: Users seeing `next_monday(thursday)` would reasonably expect it to return the following Monday, not Thursday.

## Fix

The function should either:

**Option 1: Rename the function** to better reflect its actual behavior:

```diff
diff --git a/pandas/tseries/holiday.py b/pandas/tseries/holiday.py
--- a/pandas/tseries/holiday.py
+++ b/pandas/tseries/holiday.py
-def next_monday(dt: datetime) -> datetime:
+def weekend_to_following_monday(dt: datetime) -> datetime:
     """
-    If holiday falls on Saturday, use following Monday instead;
-    if holiday falls on Sunday, use Monday instead
+    If date falls on Saturday or Sunday, use following Monday instead.
+    Otherwise, return the date unchanged (for use with holiday observance).
+
+    Parameters
+    ----------
+    dt : datetime
+        The input date
+
+    Returns
+    -------
+    datetime
+        Following Monday if weekend, otherwise the input date
     """
     if dt.weekday() == 5:
         return dt + timedelta(2)
     elif dt.weekday() == 6:
         return dt + timedelta(1)
     return dt
```

**Option 2: Fix the implementation** to match the name:

```diff
diff --git a/pandas/tseries/holiday.py b/pandas/tseries/holiday.py
--- a/pandas/tseries/holiday.py
+++ b/pandas/tseries/holiday.py
 def next_monday(dt: datetime) -> datetime:
     """
-    If holiday falls on Saturday, use following Monday instead;
-    if holiday falls on Sunday, use Monday instead
+    Return the next Monday on or after the given date.
+
+    Parameters
+    ----------
+    dt : datetime
+        The input date
+
+    Returns
+    -------
+    datetime
+        The next Monday (if dt is Monday, returns dt)
     """
-    if dt.weekday() == 5:
-        return dt + timedelta(2)
-    elif dt.weekday() == 6:
-        return dt + timedelta(1)
-    return dt
+    days_until_monday = (7 - dt.weekday()) % 7
+    if days_until_monday == 0:
+        return dt  # Already Monday
+    return dt + timedelta(days_until_monday)
```

**Recommendation**: Option 1 is safer as it preserves backward compatibility. The function appears to be used internally for holiday observance rules, so changing behavior (Option 2) could break existing code. A deprecation cycle would be needed if renaming.