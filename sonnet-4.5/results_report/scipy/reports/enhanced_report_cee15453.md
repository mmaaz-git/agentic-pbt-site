# Bug Report: scipy.io.arff DateAttribute Always-True Conditional Logic Error

**Target**: `scipy.io.arff._arffread.DateAttribute._get_date_format`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_get_date_format` method contains a logic error at line 276 where `elif "yy":` is used instead of `elif "yy" in pattern:`, causing the condition to always evaluate to True and incorrectly setting datetime_unit to 'Y' for date patterns without year components.

## Property-Based Test

```python
from scipy.io.arff._arffread import DateAttribute
from hypothesis import given, strategies as st, assume


@given(st.text(alphabet='MdHms-/: ', min_size=1, max_size=50))
def test_date_format_without_year_shouldnt_set_year_unit(pattern_body):
    """
    Test that date patterns without year components don't incorrectly
    set datetime_unit to 'Y' during processing.

    This test exposes a bug where 'elif "yy":' (always True) causes
    datetime_unit to be incorrectly set to 'Y' for patterns without years.
    """
    assume('y' not in pattern_body.lower())
    assume(any(x in pattern_body for x in ['M', 'd', 'H', 'm', 's']))

    pattern_str = f"date {pattern_body}"

    try:
        date_fmt, datetime_unit = DateAttribute._get_date_format(pattern_str)
        assert datetime_unit != "Y", \
            f"Pattern {pattern_body} has no year but datetime_unit is 'Y'"
    except ValueError:
        pass  # Some patterns may be invalid, that's OK


if __name__ == "__main__":
    # Run the test
    test_date_format_without_year_shouldnt_set_year_unit()
```

<details>

<summary>
**Failing input**: `'H'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/24
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_date_format_without_year_shouldnt_set_year_unit FAILED     [100%]

=================================== FAILURES ===================================
_____________ test_date_format_without_year_shouldnt_set_year_unit _____________

    @given(st.text(alphabet='MdHms-/: ', min_size=1, max_size=50))
>   def test_date_format_without_year_shouldnt_set_year_unit(pattern_body):
                   ^^^

hypo.py:6:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

pattern_body = 'H'

    @given(st.text(alphabet='MdHms-/: ', min_size=1, max_size=50))
    def test_date_format_without_year_shouldnt_set_year_unit(pattern_body):
        """
        Test that date patterns without year components don't incorrectly
        set datetime_unit to 'Y' during processing.

        This test exposes a bug where 'elif "yy":' (always True) causes
        datetime_unit to be incorrectly set to 'Y' for patterns without years.
        """
        assume('y' not in pattern_body.lower())
        assume(any(x in pattern_body for x in ['M', 'd', 'H', 'm', 's']))

        pattern_str = f"date {pattern_body}"

        try:
            date_fmt, datetime_unit = DateAttribute._get_date_format(pattern_str)
>           assert datetime_unit != "Y", \
                f"Pattern {pattern_body} has no year but datetime_unit is 'Y'"
E               AssertionError: Pattern H has no year but datetime_unit is 'Y'
E               assert 'Y' != 'Y'
E               Falsifying example: test_date_format_without_year_shouldnt_set_year_unit(
E                   pattern_body='H',
E               )

hypo.py:21: AssertionError
=========================== short test summary info ============================
FAILED hypo.py::test_date_format_without_year_shouldnt_set_year_unit - Assert...
============================== 1 failed in 0.22s ===============================
```
</details>

## Reproducing the Bug

```python
from scipy.io.arff._arffread import DateAttribute

# Test cases that expose the bug
test_patterns = [
    "date H",      # Single hour - should be 'h' but returns 'Y'
    "date m",      # Single minute - should be 'm' but returns 'Y'
    "date s",      # Single second - should be 's' but returns 'Y'
    "date HH",     # Double hour - correctly returns 'h'
    "date mm",     # Double minute - correctly returns 'm'
    "date ss",     # Double second - correctly returns 's'
    "date MM-dd",  # Month-day - correctly returns 'D' (bug masked)
]

print("Testing scipy.io.arff DateAttribute._get_date_format bug")
print("=" * 60)
print()
print("Bug: Line 276 contains 'elif \"yy\":' instead of 'elif \"yy\" in pattern:'")
print("This causes the condition to always evaluate to True.")
print()
print(f"Proof: bool('yy') = {bool('yy')} (always True!)")
print()
print("Test Results:")
print("-" * 60)

for pattern in test_patterns:
    try:
        result_fmt, result_unit = DateAttribute._get_date_format(pattern)
        # Determine expected unit based on pattern content
        if "H" in pattern or "HH" in pattern:
            expected = "h"
        elif "m" in pattern and "mm" not in pattern:  # single 'm'
            expected = "m"
        elif "mm" in pattern:
            expected = "m"
        elif "s" in pattern and "ss" not in pattern:  # single 's'
            expected = "s"
        elif "ss" in pattern:
            expected = "s"
        elif "dd" in pattern:
            expected = "D"
        elif "MM" in pattern:
            expected = "M"
        else:
            expected = "?"

        is_correct = result_unit == expected or (pattern in ["date MM-dd"] and result_unit == "D")
        status = "✓" if is_correct else "✗ BUG EXPOSED"

        print(f"Pattern: {pattern:<15} Result unit: {result_unit:<3} Expected: {expected:<3} {status}")
    except ValueError as e:
        print(f"Pattern: {pattern:<15} Error: {e}")

print()
print("Detailed Analysis for 'date H':")
print("-" * 60)
print("Execution flow:")
print("1. Pattern 'H' enters _get_date_format")
print("2. Line 273: 'yyyy' in 'H' -> False (skip)")
print("3. Line 276: 'yy' -> Always True (BUG!)")
print("4. Line 277: pattern.replace('yy', '%y') -> No change")
print("5. Line 278: datetime_unit = 'Y' (WRONG!)")
print("6. No HH/mm/ss checks match to overwrite")
print("7. Returns datetime_unit = 'Y' (should be 'h')")
```

<details>

<summary>
Bug demonstrates incorrect datetime_unit='Y' for patterns without year components
</summary>
```
Testing scipy.io.arff DateAttribute._get_date_format bug
============================================================

Bug: Line 276 contains 'elif "yy":' instead of 'elif "yy" in pattern:'
This causes the condition to always evaluate to True.

Proof: bool('yy') = True (always True!)

Test Results:
------------------------------------------------------------
Pattern: date H          Result unit: Y   Expected: h   ✗ BUG EXPOSED
Pattern: date m          Result unit: Y   Expected: m   ✗ BUG EXPOSED
Pattern: date s          Result unit: Y   Expected: s   ✗ BUG EXPOSED
Pattern: date HH         Result unit: h   Expected: h   ✓
Pattern: date mm         Result unit: m   Expected: m   ✓
Pattern: date ss         Result unit: s   Expected: s   ✓
Pattern: date MM-dd      Result unit: D   Expected: D   ✓

Detailed Analysis for 'date H':
------------------------------------------------------------
Execution flow:
1. Pattern 'H' enters _get_date_format
2. Line 273: 'yyyy' in 'H' -> False (skip)
3. Line 276: 'yy' -> Always True (BUG!)
4. Line 277: pattern.replace('yy', '%y') -> No change
5. Line 278: datetime_unit = 'Y' (WRONG!)
6. No HH/mm/ss checks match to overwrite
7. Returns datetime_unit = 'Y' (should be 'h')
```
</details>

## Why This Is A Bug

This is an unambiguous programming error where a conditional check `elif "yy":` always evaluates to True because any non-empty string is truthy in Python. The correct syntax should be `elif "yy" in pattern:` to check if the substring "yy" exists in the pattern string.

The bug violates the expected behavior documented in the code comments, which state the function should "convert time pattern from Java's SimpleDateFormat to C's format" and set the datetime_unit based on the actual components present in the pattern. When a pattern contains only time components (like 'H', 'm', or 's') without any year reference, the datetime_unit should reflect the time granularity ('h', 'm', or 's'), not 'Y' for year.

This error manifests specifically for single-character time patterns where the subsequent double-character checks (HH, mm, ss) don't match to overwrite the incorrectly set 'Y' value. For patterns with double-character components, the bug is masked because later conditions correctly overwrite the datetime_unit.

## Relevant Context

The DateAttribute class is part of SciPy's ARFF file format reader implementation located in `/scipy/io/arff/_arffread.py`. While the official SciPy documentation states that date type attributes are "not implemented", the code does exist and attempts to handle date parsing by converting Java SimpleDateFormat patterns to Python strptime format.

The bug occurs at line 276 in the `_get_date_format` static method. The surrounding code pattern clearly shows the intended behavior - all other conditions use the `in pattern` syntax to check for substring presence:
- Line 273: `if "yyyy" in pattern:`
- Line 279: `if "MM" in pattern:`
- Line 282: `if "dd" in pattern:`

The implementation determines the numpy datetime64 unit based on the finest granularity in the pattern (year='Y', month='M', day='D', hour='h', minute='m', second='s').

Code location: https://github.com/scipy/scipy/blob/main/scipy/io/arff/_arffread.py#L276

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