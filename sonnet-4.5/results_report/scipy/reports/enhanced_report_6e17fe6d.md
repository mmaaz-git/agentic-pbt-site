# Bug Report: scipy.io.arff DateAttribute Validation Logic Error

**Target**: `scipy.io.arff._arffread.DateAttribute._get_date_format`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `DateAttribute._get_date_format` method incorrectly accepts invalid date format patterns due to a logic error on line 276 where `elif "yy":` always evaluates to True, causing the function to incorrectly set `datetime_unit = "Y"` even when no year component exists in the pattern, thereby bypassing the validation check on line 298.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from scipy.io.arff._arffread import DateAttribute
import pytest


@given(st.text(alphabet='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=1, max_size=10))
def test_invalid_date_format_should_raise_valueerror(invalid_pattern):
    assume('yyyy' not in invalid_pattern)
    assume('yy' not in invalid_pattern)
    assume('MM' not in invalid_pattern)
    assume('dd' not in invalid_pattern)
    assume('HH' not in invalid_pattern)
    assume('mm' not in invalid_pattern)
    assume('ss' not in invalid_pattern)

    date_string = f'date "{invalid_pattern}"'

    with pytest.raises(ValueError):
        DateAttribute._get_date_format(date_string)

if __name__ == "__main__":
    test_invalid_date_format_should_raise_valueerror()
```

<details>

<summary>
**Failing input**: `invalid_pattern='0'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/43
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_invalid_date_format_should_raise_valueerror FAILED         [100%]

=================================== FAILURES ===================================
_______________ test_invalid_date_format_should_raise_valueerror _______________

    @given(st.text(alphabet='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=1, max_size=10))
>   def test_invalid_date_format_should_raise_valueerror(invalid_pattern):
                   ^^^

hypo.py:7:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

invalid_pattern = '0'

    @given(st.text(alphabet='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=1, max_size=10))
    def test_invalid_date_format_should_raise_valueerror(invalid_pattern):
        assume('yyyy' not in invalid_pattern)
        assume('yy' not in invalid_pattern)
        assume('MM' not in invalid_pattern)
        assume('dd' not in invalid_pattern)
        assume('HH' not in invalid_pattern)
        assume('mm' not in invalid_pattern)
        assume('ss' not in invalid_pattern)

        date_string = f'date "{invalid_pattern}"'

>       with pytest.raises(ValueError):
             ^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE <class 'ValueError'>
E       Falsifying example: test_invalid_date_format_should_raise_valueerror(
E           invalid_pattern='0',
E       )
E       Explanation:
E           These lines were always and only run by failing examples:
E               /home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py:298

hypo.py:18: Failed
=========================== short test summary info ============================
FAILED hypo.py::test_invalid_date_format_should_raise_valueerror - Failed: DI...
============================== 1 failed in 0.31s ===============================
```
</details>

## Reproducing the Bug

```python
from scipy.io.arff._arffread import DateAttribute

# Test with an invalid date pattern that contains no valid date components
invalid_pattern = 'date "abc"'

try:
    result_pattern, result_unit = DateAttribute._get_date_format(invalid_pattern)
    print(f"BUG: Invalid pattern '{invalid_pattern}' was accepted")
    print(f"Returned: pattern='{result_pattern}', unit='{result_unit}'")
    print(f"Expected: Should raise ValueError")
except ValueError as e:
    print(f"Correct: ValueError raised - {e}")
```

<details>

<summary>
BUG: Invalid pattern was incorrectly accepted
</summary>
```
BUG: Invalid pattern 'date "abc"' was accepted
Returned: pattern='abc', unit='Y'
Expected: Should raise ValueError
```
</details>

## Why This Is A Bug

The `_get_date_format` method is designed to parse and validate Java SimpleDateFormat patterns from ARFF date attribute declarations and convert them to Python datetime format strings. The function contains explicit validation logic on lines 298-299 that should raise a `ValueError` when `datetime_unit` is `None`, indicating no valid date components were found in the pattern.

However, due to the logic error on line 276, the condition `elif "yy":` always evaluates to `True` because the string literal `"yy"` is a truthy value in Python. This means that for any pattern that doesn't contain "yyyy", the code will always execute the block that sets `datetime_unit = "Y"`, even when the pattern contains no year component at all. This prevents the validation check from ever detecting invalid patterns that lack all date components.

The ARFF specification requires date attributes to have valid date format patterns following Java's SimpleDateFormat conventions. Accepting invalid patterns like "abc" or "0" violates this specification and could lead to unexpected behavior when attempting to parse date values later.

## Relevant Context

The bug is located in `/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py` at line 276. The function is part of scipy's ARFF file reader implementation, which parses the Attribute-Relation File Format commonly used in machine learning datasets.

The code attempts to convert Java SimpleDateFormat patterns (used in ARFF files) to Python's strftime format codes. Valid patterns should contain at least one of: yyyy/yy (year), MM (month), dd (day), HH (hour), mm (minute), or ss (second).

While scipy's documentation mentions that date attributes are "not implemented", the code exists and contains validation logic that should work correctly. The bug allows malformed ARFF files to be accepted when they should be rejected according to the ARFF specification.

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