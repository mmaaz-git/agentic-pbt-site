# Bug Report: scipy.io.arff DateAttribute Fails to Validate Empty Format Strings

**Target**: `scipy.io.arff._arffread.DateAttribute._get_date_format`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `DateAttribute._get_date_format` method contains a logic error on line 276 where `elif "yy":` always evaluates to `True`, preventing proper validation of empty or invalid date format patterns and causing them to be incorrectly accepted with `datetime_unit='Y'`.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis to find the bug in
scipy.io.arff DateAttribute format validation.
"""

from hypothesis import given, strategies as st, settings
from scipy.io.arff._arffread import DateAttribute
import pytest


@given(st.text(alphabet=' \t', min_size=0, max_size=10))
@settings(max_examples=20)  # Reduced for demonstration
def test_date_format_empty_or_whitespace_bug(whitespace):
    """
    Test that DateAttribute raises ValueError for empty or whitespace-only
    date format patterns.
    """
    pattern = f"date '{whitespace}'"
    try:
        attr = DateAttribute.parse_attribute('test', pattern)
        # If we get here without exception, the test found the bug
        print(f"FAILED: Pattern '{pattern}' should have raised ValueError")
        print(f"  Got: datetime_unit={attr.datetime_unit}, date_format='{attr.date_format}'")
        pytest.fail(f"Should have raised ValueError for pattern '{pattern}', "
                   f"but got datetime_unit={attr.datetime_unit}")
    except ValueError:
        # This is the expected behavior
        print(f"OK: Pattern '{pattern}' correctly raised ValueError")


if __name__ == "__main__":
    print("Running property-based test to find date format validation bug...")
    print("=" * 60)
    test_date_format_empty_or_whitespace_bug()
```

<details>

<summary>
**Failing input**: `whitespace=''`
</summary>
```
Running property-based test to find date format validation bug...
============================================================
FAILED: Pattern 'date ''' should have raised ValueError
  Got: datetime_unit=Y, date_format='''
FAILED: Pattern 'date ' 		 			  	'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '   	 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '		  					 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date ' 		   			 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '		 				 		'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date ' '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '		  	   	 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '		 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '				 		   '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date ''' should have raised ValueError
  Got: datetime_unit=Y, date_format='''
FAILED: Pattern 'date ' 		  			 	'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	  		  	'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '  	  	'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '  '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '   	   	 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '  		'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date ' 	'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '		  	'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '		 		 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date ' 	     		 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date ' 		 		'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '		   		'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	  	  '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '        	 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	 	'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '  	   				'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '		  	 	   '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date ' 			 		 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '			   				'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date ' 			 	 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	   			'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '		  	 				'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	     	 	 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	  		 				'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date ' 	  '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '  				 	'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '			 	   		'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '   '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	 	  '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '  	 	 	'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date ' 			 	  	 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date ' 		 				  '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '   	 	'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	 	 				  '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '		 			  	 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '		  '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '			  		 		'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date ' 	   '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date ' 	  	 				'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '		'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	  		 			 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	 		 	  		'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '  	 			'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '		     		 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '							'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '     	  		'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '				 	 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	   		'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '				 		  	'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '			 	 	 		'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '  			  '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '    		 		'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	    			  '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '  	 	 '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	  	  		 	'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date '	  '' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date ' 			'' should have raised ValueError
  Got: datetime_unit=Y, date_format=''
FAILED: Pattern 'date ''' should have raised ValueError
  Got: datetime_unit=Y, date_format='''
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 35, in <module>
    test_date_format_empty_or_whitespace_bug()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 13, in test_date_format_empty_or_whitespace_bug
    @settings(max_examples=20)  # Reduced for demonstration
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/27/hypo.py", line 25, in test_date_format_empty_or_whitespace_bug
    pytest.fail(f"Should have raised ValueError for pattern '{pattern}', "
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
               f"but got datetime_unit={attr.datetime_unit}")
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: Should have raised ValueError for pattern 'date ''', but got datetime_unit=Y
Falsifying example: test_date_format_empty_or_whitespace_bug(
    whitespace='',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Demonstration of the scipy.io.arff DateAttribute bug.
This script shows that empty or whitespace-only date format patterns
do not raise ValueError as expected.
"""

from scipy.io.arff._arffread import DateAttribute

# Test 1: Empty date format string
print("Test 1: Empty date format string")
print("=" * 50)
try:
    attr = DateAttribute.parse_attribute('test_attr', "date ''")
    print(f"Expected: ValueError should be raised")
    print(f"Actual: No error raised!")
    print(f"  - datetime_unit = {attr.datetime_unit}")
    print(f"  - date_format = '{attr.date_format}'")
except ValueError as e:
    print(f"ValueError correctly raised: {e}")
print()

# Test 2: Whitespace-only date format string
print("Test 2: Whitespace-only date format string")
print("=" * 50)
try:
    attr = DateAttribute.parse_attribute('test_attr', "date '   '")
    print(f"Expected: ValueError should be raised")
    print(f"Actual: No error raised!")
    print(f"  - datetime_unit = {attr.datetime_unit}")
    print(f"  - date_format = '{attr.date_format}'")
except ValueError as e:
    print(f"ValueError correctly raised: {e}")
print()

# Test 3: Tab character only
print("Test 3: Tab character only")
print("=" * 50)
try:
    attr = DateAttribute.parse_attribute('test_attr', "date '\t'")
    print(f"Expected: ValueError should be raised")
    print(f"Actual: No error raised!")
    print(f"  - datetime_unit = {attr.datetime_unit}")
    print(f"  - date_format = '{attr.date_format}'")
except ValueError as e:
    print(f"ValueError correctly raised: {e}")
print()

# Test 4: Valid date format for comparison
print("Test 4: Valid date format (for comparison)")
print("=" * 50)
try:
    attr = DateAttribute.parse_attribute('test_attr', "date 'yyyy-MM-dd'")
    print(f"Valid format correctly parsed:")
    print(f"  - datetime_unit = {attr.datetime_unit}")
    print(f"  - date_format = '{attr.date_format}'")
except ValueError as e:
    print(f"Unexpected error: {e}")
```

<details>

<summary>
Empty date format incorrectly accepted with datetime_unit='Y'
</summary>
```
Test 1: Empty date format string
==================================================
Expected: ValueError should be raised
Actual: No error raised!
  - datetime_unit = Y
  - date_format = '''

Test 2: Whitespace-only date format string
==================================================
Expected: ValueError should be raised
Actual: No error raised!
  - datetime_unit = Y
  - date_format = ''

Test 3: Tab character only
==================================================
Expected: ValueError should be raised
Actual: No error raised!
  - datetime_unit = Y
  - date_format = ''

Test 4: Valid date format (for comparison)
==================================================
Valid format correctly parsed:
  - datetime_unit = D
  - date_format = '%Y-%m-%d'
```
</details>

## Why This Is A Bug

This bug violates the intended validation logic in the `_get_date_format` method. The code at lines 298-299 is designed to raise a `ValueError` with the message "Invalid or unsupported date format" when no valid date components are found in the pattern (i.e., when `datetime_unit` remains `None`).

However, due to the logic error on line 276 where `elif "yy":` is always `True` (because the string `"yy"` is truthy in Python), the code unconditionally sets `datetime_unit = "Y"` for any pattern that doesn't contain `"yyyy"`. This prevents the validation check from working correctly, allowing empty or whitespace-only date format patterns to be accepted when they should be rejected.

The bug affects the public `loadarff()` API, which successfully loads ARFF files with empty date formats instead of raising the appropriate error. This could lead to confusion and downstream errors when trying to parse actual date values with these invalid formats.

## Relevant Context

The bug is located in `/scipy/io/arff/_arffread.py` at line 276. The ARFF format is commonly used in machine learning applications, particularly with the Weka toolkit. While the SciPy documentation mentions that date attributes have limited support, the code is functional and accessible through the public API.

Code location: https://github.com/scipy/scipy/blob/main/scipy/io/arff/_arffread.py#L276

The DateAttribute class converts Java's SimpleDateFormat patterns to Python's strftime format. The bug specifically affects the pattern validation logic that should ensure at least one valid date component (year, month, day, hour, minute, or second) is present in the format string.

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