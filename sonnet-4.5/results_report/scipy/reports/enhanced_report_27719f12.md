# Bug Report: scipy.io.arff DateAttribute._get_date_format Always True Conditional

**Target**: `scipy.io.arff._arffread.DateAttribute._get_date_format`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_get_date_format` method contains a logic error where `elif "yy":` on line 276 always evaluates to True, causing invalid date patterns to be incorrectly accepted and returning 'Y' as the datetime unit instead of raising a ValueError.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from scipy.io.arff._arffread import DateAttribute

@given(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=20))
@settings(max_examples=100)
def test_date_format_invalid_patterns_should_raise(pattern):
    valid_components = ['yyyy', 'yy', 'MM', 'dd', 'HH', 'mm', 'ss']
    assume(not any(comp in pattern for comp in valid_components))
    assume('z' not in pattern.lower() and 'Z' not in pattern)

    try:
        result_pattern, unit = DateAttribute._get_date_format(f"date {pattern}")

        print(f"FAILED: Pattern '{pattern}' has no valid date components but returned "
              f"result='{result_pattern}', unit='{unit}' instead of raising ValueError. "
              f"This is due to bug on line 276: 'elif \"yy\":' which is always True")
        return False
    except ValueError:
        # This is the expected behavior
        return True

# Run the test
test_date_format_invalid_patterns_should_raise()
```

<details>

<summary>
**Failing input**: `pattern='A'`
</summary>
```
FAILED: Pattern 'A' has no valid date components but returned result='A', unit='Y' instead of raising ValueError. This is due to bug on line 276: 'elif "yy":' which is always True
You can add @seed(2971442250046007956709499889635078536) to this test to reproduce this failure.
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 23, in <module>
    test_date_format_invalid_patterns_should_raise()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 5, in test_date_format_invalid_patterns_should_raise
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/internal/healthcheck.py", line 21, in fail_health_check
    raise FailedHealthCheck(message)
hypothesis.errors.FailedHealthCheck: Tests run under @given should return None, but test_date_format_invalid_patterns_should_raise returned False instead.
```
</details>

## Reproducing the Bug

```python
from scipy.io.arff._arffread import DateAttribute

# Test with a completely invalid pattern that has no date components
pattern = "A"
try:
    result_pattern, unit = DateAttribute._get_date_format(f"date {pattern}")
    print(f"Pattern: {pattern}")
    print(f"Result: {result_pattern}")
    print(f"Unit: {unit}")
    print("ERROR: Should have raised ValueError for invalid pattern")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")
```

<details>

<summary>
Invalid pattern accepted with incorrect datetime unit
</summary>
```
Pattern: A
Result: A
Unit: Y
ERROR: Should have raised ValueError for invalid pattern
```
</details>

## Why This Is A Bug

This violates expected behavior because the code on line 276 checks `elif "yy":` instead of `elif "yy" in pattern:`. Since the string literal "yy" is truthy, this condition always evaluates to True when reached, causing:

1. **Invalid patterns are accepted**: Patterns without any valid date components (yyyy, yy, MM, dd, HH, mm, ss) don't raise the expected ValueError
2. **Incorrect datetime unit**: The unit is always set to 'Y' (year) even for patterns with no year component
3. **Error handling bypassed**: The validation check on lines 298-299 never triggers because datetime_unit is always set

This contradicts the ARFF specification which is based on Java's SimpleDateFormat, where invalid patterns should throw an exception. The code already has appropriate error handling (`ValueError("Invalid or unsupported date format")`) that should be triggered but isn't due to this bug.

## Relevant Context

The SciPy documentation explicitly lists date type attributes under "Not implemented functionality" for scipy.io.arff.loadarff, indicating this feature is experimental/unsupported. However, the code attempts to implement date handling with validation logic.

The ARFF specification follows Java SimpleDateFormat patterns where:
- Valid components include: yyyy, yy, MM, dd, HH, mm, ss
- Invalid patterns should raise an IllegalArgumentException (ValueError in Python)

The bug only manifests when:
- The pattern doesn't contain "yyyy" (line 273 check fails)
- The pattern reaches the buggy `elif "yy":` check on line 276
- Patterns with MM, dd, HH, mm, or ss may mask the bug by overwriting datetime_unit

Code location: `/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py:276`

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