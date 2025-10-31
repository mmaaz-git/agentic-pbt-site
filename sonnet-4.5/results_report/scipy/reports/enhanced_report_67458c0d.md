# Bug Report: scipy.io.arff DateAttribute Always-True Condition Allows Invalid Date Patterns

**Target**: `scipy.io.arff._arffread.DateAttribute._get_date_format`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_get_date_format` method contains a logic error at line 276 where `elif "yy":` always evaluates to True, causing invalid date patterns without any date/time components to be incorrectly accepted with `datetime_unit="Y"` instead of raising a ValueError.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from scipy.io.arff._arffread import DateAttribute


@given(st.text().filter(lambda x: 'yyyy' not in x.lower() and 'yy' not in x.lower()
                                     and 'mm' not in x.lower() and 'dd' not in x.lower()
                                     and 'hh' not in x.lower() and 'ss' not in x.lower()))
@settings(max_examples=100)
def test_date_format_no_components_should_fail(text):
    assume(len(text.strip()) > 0)

    try:
        pattern, unit = DateAttribute._get_date_format(f"date {text}")
        assert False, f"Should have raised ValueError, but got unit={unit}"
    except ValueError:
        pass

if __name__ == "__main__":
    test_date_format_no_components_should_fail()
```

<details>

<summary>
**Failing input**: `'0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 19, in <module>
    test_date_format_no_components_should_fail()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 6, in test_date_format_no_components_should_fail
    and 'mm' not in x.lower() and 'dd' not in x.lower()

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 14, in test_date_format_no_components_should_fail
    assert False, f"Should have raised ValueError, but got unit={unit}"
           ^^^^^
AssertionError: Should have raised ValueError, but got unit=Y
Falsifying example: test_date_format_no_components_should_fail(
    text='0',
)
```
</details>

## Reproducing the Bug

```python
from scipy.io.arff._arffread import DateAttribute

# Test case 1: Pattern with no date components
try:
    pattern, unit = DateAttribute._get_date_format("date 'just text'")
    print(f"Pattern: {pattern}, Unit: {unit}")
    print("ERROR: Should have raised ValueError for pattern with no date components")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")

# Test case 2: Pattern with only numbers but no date components
try:
    pattern, unit = DateAttribute._get_date_format("date '12345'")
    print(f"Pattern: {pattern}, Unit: {unit}")
    print("ERROR: Should have raised ValueError for pattern with no date components")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")

# Test case 3: Empty pattern (after stripping quotes)
try:
    pattern, unit = DateAttribute._get_date_format("date ''")
    print(f"Pattern: {pattern}, Unit: {unit}")
    print("ERROR: Should have raised ValueError for empty pattern")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")

# Test case 4: Pattern with valid component (yyyy) - should work correctly
try:
    pattern, unit = DateAttribute._get_date_format("date 'yyyy-MM-dd'")
    print(f"Pattern (with yyyy): {pattern}, Unit: {unit}")
except ValueError as e:
    print(f"Unexpected error for valid pattern: {e}")

# Test case 5: Pattern with valid component (yy) - should work correctly
try:
    pattern, unit = DateAttribute._get_date_format("date 'yy-MM-dd'")
    print(f"Pattern (with yy): {pattern}, Unit: {unit}")
except ValueError as e:
    print(f"Unexpected error for valid pattern: {e}")
```

<details>

<summary>
Invalid patterns incorrectly accepted with Unit=Y
</summary>
```
Pattern: just text, Unit: Y
ERROR: Should have raised ValueError for pattern with no date components
Pattern: 12345, Unit: Y
ERROR: Should have raised ValueError for pattern with no date components
Pattern: ', Unit: Y
ERROR: Should have raised ValueError for empty pattern
Pattern (with yyyy): %Y-%m-%d, Unit: D
Pattern (with yy): %y-%m-%d, Unit: D
```
</details>

## Why This Is A Bug

The bug occurs at line 276 in `/scipy/io/arff/_arffread.py` where the code uses `elif "yy":` instead of `elif "yy" in pattern:`. Since the string literal `"yy"` is non-empty, this condition always evaluates to True in Python, causing the elif block to execute for every pattern that doesn't contain "yyyy" (regardless of whether it contains "yy").

This violates the function's intended behavior in several ways:
1. **Breaks error handling contract**: Lines 298-299 explicitly check if `datetime_unit is None` and raise `ValueError("Invalid or unsupported date format")` for patterns with no recognized components. The bug prevents this error from ever being raised.
2. **Inconsistent with surrounding code**: All other pattern checks use the correct syntax (`if "MM" in pattern:`, `if "dd" in pattern:`, etc. on lines 279, 282, 285, 288, 291).
3. **Allows invalid ARFF files**: The ARFF specification requires date patterns to follow Java's SimpleDateFormat, which expects valid date/time components. Patterns like "just text" or "12345" are not valid SimpleDateFormat patterns.

## Relevant Context

The ARFF (Attribute-Relation File Format) specification uses Java's SimpleDateFormat for date patterns. The default format is ISO-8601: "yyyy-MM-dd'T'HH:mm:ss". The `_get_date_format` method converts these Java patterns to Python's strftime format.

The function uses a regex `r_date = re.compile(r"[Dd][Aa][Tt][Ee]\s+[\"']?(.+?)[\"']?$")` to extract the date pattern from ARFF attribute declarations. It then processes the pattern to replace Java format codes with Python equivalents while tracking the most specific datetime unit found.

According to the SciPy documentation, date type attributes are "not implemented" in the loadarff function, suggesting this functionality may be experimental or partial. However, the code clearly shows deliberate error handling that should reject invalid patterns.

Relevant code location: `/scipy/io/arff/_arffread.py:276`
Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.loadarff.html

## Proposed Fix

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -273,7 +273,7 @@ class DateAttribute(Attribute):
             if "yyyy" in pattern:
                 pattern = pattern.replace("yyyy", "%Y")
                 datetime_unit = "Y"
-            elif "yy":
+            elif "yy" in pattern:
                 pattern = pattern.replace("yy", "%y")
                 datetime_unit = "Y"
             if "MM" in pattern:
```