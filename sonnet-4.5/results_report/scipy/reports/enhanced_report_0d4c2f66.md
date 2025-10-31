# Bug Report: scipy.io.arff DateAttribute._get_date_format Always-True Condition Prevents Error Handling

**Target**: `scipy.io.arff._arffread.DateAttribute._get_date_format`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `DateAttribute._get_date_format` method has an always-true condition `elif "yy":` on line 276 that should be `elif "yy" in pattern:`. This causes the method to incorrectly accept invalid date patterns instead of raising `ValueError`.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for scipy.io.arff DateAttribute bug."""

from hypothesis import given, strategies as st, example
from scipy.io.arff._arffread import DateAttribute

@given(st.sampled_from([
    "date ''",           # Empty pattern - should raise ValueError
    "date 'XXX'",        # Invalid pattern - should raise ValueError
    "date 'abc'",        # Invalid pattern - should raise ValueError
    "date '123'",        # Invalid pattern - should raise ValueError
]))
@example("date ''")
def test_invalid_date_formats_raise_error(date_string):
    """
    Test that invalid date formats with no valid date/time components raise ValueError.
    Due to the bug at line 276, these invalid patterns are incorrectly accepted.
    """
    pattern = date_string.split("'")[1] if "'" in date_string else ""

    # Check if pattern contains any valid date/time component
    valid_components = ['yyyy', 'yy', 'MM', 'dd', 'HH', 'mm', 'ss']
    has_valid_component = any(comp in pattern for comp in valid_components)

    if not has_valid_component:
        # Should raise ValueError for invalid patterns
        try:
            date_format, datetime_unit = DateAttribute._get_date_format(date_string)
            # If we get here without exception, the bug is present
            assert False, f"Expected ValueError for invalid pattern '{pattern}', but got Format: {date_format}, Unit: {datetime_unit}"
        except ValueError as e:
            # This is the expected behavior
            assert "Invalid or unsupported date format" in str(e)

if __name__ == "__main__":
    test_invalid_date_formats_raise_error()
```

<details>

<summary>
**Failing input**: `"date ''"`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 36, in <module>
    test_invalid_date_formats_raise_error()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 8, in test_invalid_date_formats_raise_error
    "date ''",           # Empty pattern - should raise ValueError
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 30, in test_invalid_date_formats_raise_error
    assert False, f"Expected ValueError for invalid pattern '{pattern}', but got Format: {date_format}, Unit: {datetime_unit}"
           ^^^^^
AssertionError: Expected ValueError for invalid pattern '', but got Format: ', Unit: Y
Falsifying explicit example: test_invalid_date_formats_raise_error(
    date_string="date ''",
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction case for scipy.io.arff DateAttribute bug."""

from scipy.io.arff._arffread import DateAttribute

# Test case 1: Pattern with only month (should have unit='M')
print("Test 1: date 'MM'")
date_format, datetime_unit = DateAttribute._get_date_format("date 'MM'")
print(f"Format: {date_format}, Unit: {datetime_unit}")
print(f"Expected: Format: %m, Unit: M")
print(f"Bug causes unit to be incorrectly 'Y' instead of 'M'? {datetime_unit == 'Y'}")
print()

# Test case 2: Empty pattern (should raise ValueError)
print("Test 2: date ''")
try:
    date_format, datetime_unit = DateAttribute._get_date_format("date ''")
    print(f"Format: {date_format}, Unit: {datetime_unit}")
    print("ERROR: Should have raised ValueError for invalid format")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")
print()

# Test case 3: Invalid pattern with no date components
print("Test 3: date 'XXX'")
try:
    date_format, datetime_unit = DateAttribute._get_date_format("date 'XXX'")
    print(f"Format: {date_format}, Unit: {datetime_unit}")
    print("ERROR: Should have raised ValueError for invalid format")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")
print()

# Test case 4: Pattern with day only
print("Test 4: date 'dd'")
date_format, datetime_unit = DateAttribute._get_date_format("date 'dd'")
print(f"Format: {date_format}, Unit: {datetime_unit}")
print(f"Expected: Format: %d, Unit: D")
print()

# Test case 5: Pattern with time components only
print("Test 5: date 'HH:mm:ss'")
date_format, datetime_unit = DateAttribute._get_date_format("date 'HH:mm:ss'")
print(f"Format: {date_format}, Unit: {datetime_unit}")
print(f"Expected: Format: %H:%M:%S, Unit: s")
print()

# Test case 6: Complex pattern without year
print("Test 6: date 'MM-dd HH:mm'")
date_format, datetime_unit = DateAttribute._get_date_format("date 'MM-dd HH:mm'")
print(f"Format: {date_format}, Unit: {datetime_unit}")
print(f"Expected: Format: %m-%d %H:%M, Unit: m")
```

<details>

<summary>
Invalid date patterns incorrectly accepted instead of raising ValueError
</summary>
```
Test 1: date 'MM'
Format: %m, Unit: M
Expected: Format: %m, Unit: M
Bug causes unit to be incorrectly 'Y' instead of 'M'? False

Test 2: date ''
Format: ', Unit: Y
ERROR: Should have raised ValueError for invalid format

Test 3: date 'XXX'
Format: XXX, Unit: Y
ERROR: Should have raised ValueError for invalid format

Test 4: date 'dd'
Format: %d, Unit: D
Expected: Format: %d, Unit: D

Test 5: date 'HH:mm:ss'
Format: %H:%M:%S, Unit: s
Expected: Format: %H:%M:%S, Unit: s

Test 6: date 'MM-dd HH:mm'
Format: %m-%d %H:%M, Unit: m
Expected: Format: %m-%d %H:%M, Unit: m
```
</details>

## Why This Is A Bug

The condition `elif "yy":` on line 276 is always true because `"yy"` is a non-empty string, which is truthy in Python. This violates the expected behavior in several ways:

1. **Breaks Error Handling**: The method is designed to raise `ValueError("Invalid or unsupported date format")` when `datetime_unit` remains `None` (lines 298-299). However, the always-true condition sets `datetime_unit = "Y"` for all patterns without "yyyy", preventing proper error detection.

2. **Violates ARFF Specification**: According to the ARFF format specification (Waikato/WEKA), date patterns must follow Java's SimpleDateFormat. Invalid patterns should be rejected, not silently accepted.

3. **Contradicts Code Intent**: The clear intent is `elif "yy" in pattern:` to check if the pattern contains a two-digit year. The missing `in pattern` check is an obvious typo that changes the logic fundamentally.

4. **Causes Unnecessary Operations**: For every pattern without "yyyy", the code executes `pattern.replace("yy", "%y")` even when "yy" isn't present, which is inefficient and confusing.

## Relevant Context

The scipy.io.arff documentation explicitly states that date type attributes are "not implemented", yet the DateAttribute class exists and is partially functional. This bug affects the error handling of this partially-implemented feature.

The `_get_date_format` method converts Java SimpleDateFormat patterns to Python strftime format and determines the appropriate numpy datetime64 unit based on precision:
- Line 273-278: Handles year patterns (where the bug occurs)
- Line 279-296: Handles month, day, hour, minute, second patterns
- Line 298-299: Should raise ValueError if no valid components found

Code location: `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/io/arff/_arffread.py:276`

## Proposed Fix

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -273,7 +273,7 @@ class DateAttribute(Attribute):
         if "yyyy" in pattern:
             pattern = pattern.replace("yyyy", "%Y")
             datetime_unit = "Y"
-        elif "yy":
+        elif "yy" in pattern:
             pattern = pattern.replace("yy", "%y")
             datetime_unit = "Y"
         if "MM" in pattern:
```