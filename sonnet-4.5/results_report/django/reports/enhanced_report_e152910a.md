# Bug Report: django.template.defaultfilters.get_digit ValueError When Accessing Minus Sign in Negative Numbers

**Target**: `django.template.defaultfilters.get_digit`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_digit` filter crashes with `ValueError: invalid literal for int() with base 10: '-'` when given a negative number and a position that accesses the minus sign character, violating its documented behavior of failing silently.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for django.template.defaultfilters.get_digit with negative numbers."""

import sys
import os

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
django.setup()

from hypothesis import given, strategies as st
from django.template import defaultfilters


@given(st.integers(min_value=-1000, max_value=-1), st.integers(min_value=1, max_value=10))
def test_get_digit_negative_numbers(value, position):
    result = defaultfilters.get_digit(value, position)
    value_str = str(value)
    try:
        expected = int(value_str[-position])
    except (IndexError, ValueError):
        expected = 0
    assert result == expected or result == value


if __name__ == "__main__":
    # Run the test
    test_get_digit_negative_numbers()
```

<details>

<summary>
**Failing input**: `test_get_digit_negative_numbers(value=-1, position=2)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 33, in <module>
    test_get_digit_negative_numbers()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 21, in test_get_digit_negative_numbers
    def test_get_digit_negative_numbers(value, position):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 22, in test_get_digit_negative_numbers
    result = defaultfilters.get_digit(value, position)
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/template/defaultfilters.py", line 768, in get_digit
    return int(str(value)[-arg])
ValueError: invalid literal for int() with base 10: '-'
Falsifying example: test_get_digit_negative_numbers(
    value=-1,
    position=2,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of the get_digit bug with negative numbers."""

import sys
import os

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
django.setup()

from django.template import defaultfilters

# Test case 1: get_digit(-123, 4)
print("Test case 1: get_digit(-123, 4)")
print("Expected: Should return 0 or -123 (original value) based on documented behavior")
print("String representation: str(-123) = '-123' (4 characters)")
print("Accessing position -4: str(-123)[-4] = '-'")
print("Attempting to execute...")

try:
    result = defaultfilters.get_digit(-123, 4)
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test case 2: get_digit(-1, 2)
print("Test case 2: get_digit(-1, 2)")
print("Expected: Should return 0 or -1 (original value) based on documented behavior")
print("String representation: str(-1) = '-1' (2 characters)")
print("Accessing position -2: str(-1)[-2] = '-'")
print("Attempting to execute...")

try:
    result = defaultfilters.get_digit(-1, 2)
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError: invalid literal for int() with base 10: '-'
</summary>
```
Test case 1: get_digit(-123, 4)
Expected: Should return 0 or -123 (original value) based on documented behavior
String representation: str(-123) = '-123' (4 characters)
Accessing position -4: str(-123)[-4] = '-'
Attempting to execute...
ERROR: ValueError: invalid literal for int() with base 10: '-'

==================================================

Test case 2: get_digit(-1, 2)
Expected: Should return 0 or -1 (original value) based on documented behavior
String representation: str(-1) = '-1' (2 characters)
Accessing position -2: str(-1)[-2] = '-'
Attempting to execute...
ERROR: ValueError: invalid literal for int() with base 10: '-'
```
</details>

## Why This Is A Bug

The function's docstring explicitly states: "Return the original value for invalid input (if input or argument is not an integer, or if argument is less than 1). Otherwise, output is always an integer." This establishes a contract that the function should fail silently by returning either the original value or an integer, never raising an exception during normal operation.

The bug occurs because when accessing a negative number's string representation at a position that corresponds to the minus sign (e.g., `str(-123)[-4]` returns `"-"`), the function attempts `int("-")` which raises a `ValueError`. The current implementation only catches `IndexError` for out-of-range positions (line 769) but doesn't handle `ValueError` for non-digit characters.

This violates the documented behavior in several ways:
1. The function raises an exception instead of failing silently
2. Negative integers are valid "whole numbers" in the programming context and are already accepted by the function for other positions
3. The function successfully processes negative numbers when accessing actual digit positions (e.g., `get_digit(-123, 1)` returns `3`)
4. The inconsistent error handling (catching `IndexError` but not `ValueError`) creates unexpected behavior

## Relevant Context

The implementation is located in `django/template/defaultfilters.py` lines 753-770. The function already has robust error handling for invalid input types (lines 760-764) and out-of-range positions (lines 769-770), but misses this specific edge case.

This bug affects Django template rendering, which is user-facing. When templates use the `get_digit` filter with dynamic values that might be negative, this crash could cause template rendering failures in production.

Documentation: The function is a Django template filter commonly used in templates like `{{ value|get_digit:2 }}` to extract specific digits from numbers for display purposes.

## Proposed Fix

```diff
--- a/django/template/defaultfilters.py
+++ b/django/template/defaultfilters.py
@@ -765,7 +765,7 @@ def get_digit(value, arg):
     if arg < 1:
         return value
     try:
         return int(str(value)[-arg])
-    except IndexError:
+    except (IndexError, ValueError):
         return 0
```