# Bug Report: django.template.Variable Incorrectly Accepts Numeric Literals with Trailing Dots

**Target**: `django.template.Variable.__init__`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Variable class fails to reject numeric strings with trailing dots (e.g., "2.", "10.") due to improper exception handling, despite explicit code comments stating these inputs are invalid.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pytest
from django.template import Variable

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=1, max_value=1e10))
def test_variable_float_with_trailing_dot_should_be_rejected(num):
    """
    Property: Floats with trailing dots should be rejected as invalid.
    Evidence: Code comment on line 824 says '"2." is invalid' and code
    explicitly raises ValueError for this case on line 826.
    """
    var_str = f"{int(num)}."

    with pytest.raises((ValueError, Exception)):
        var = Variable(var_str)

# Run the test
if __name__ == "__main__":
    test_variable_float_with_trailing_dot_should_be_rejected()
```

<details>

<summary>
**Failing input**: `num=1.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 19, in <module>
    test_variable_float_with_trailing_dot_should_be_rejected()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 6, in test_variable_float_with_trailing_dot_should_be_rejected
    def test_variable_float_with_trailing_dot_should_be_rejected(num):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 14, in test_variable_float_with_trailing_dot_should_be_rejected
    with pytest.raises((ValueError, Exception)):
         ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 710, in __exit__
    fail(f"DID NOT RAISE any of {self.expected_exceptions!r}")
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: DID NOT RAISE any of (<class 'ValueError'>, <class 'Exception'>)
Falsifying example: test_variable_float_with_trailing_dot_should_be_rejected(
    num=1.0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from django.template import Variable

# Test case that should raise an error but doesn't
var = Variable("2.")

print(f"Variable created successfully for '2.'")
print(f"literal: {var.literal}")
print(f"lookups: {var.lookups}")
print()

# Try with more trailing dot cases
test_cases = ["1.", "10.", "999.", "1234."]
for test_str in test_cases:
    var = Variable(test_str)
    print(f"Variable('{test_str}'): literal={var.literal}, lookups={var.lookups}")
```

<details>

<summary>
Variable incorrectly accepts trailing dots and creates malformed objects
</summary>
```
Variable created successfully for '2.'
literal: 2.0
lookups: ('2', '')

Variable('1.'): literal=1.0, lookups=('1', '')
Variable('10.'): literal=10.0, lookups=('10', '')
Variable('999.'): literal=999.0, lookups=('999', '')
Variable('1234.'): literal=1234.0, lookups=('1234', '')
```
</details>

## Why This Is A Bug

The Variable class explicitly documents that numeric strings with trailing dots should be invalid. On line 824 of django/template/base.py, there's a comment stating `# "2." is invalid`, followed immediately by code that checks for a trailing dot and raises ValueError (lines 825-826). However, this ValueError is incorrectly caught by the outer exception handler starting at line 829, which is intended to handle non-numeric variables.

When the ValueError is caught, the code treats "2." as a variable name and splits it on the dot separator (line 848), creating a malformed Variable object with:
1. `literal` set to the float value (2.0)
2. `lookups` set to a tuple containing the integer part and an empty string ('2', '')

This violates the documented contract in three ways:
1. The input "2." should be rejected outright as stated in the comment
2. A Variable object should have either `literal` OR `lookups` set, never both
3. The lookups tuple should not contain empty strings as lookup components

The VARIABLE_ATTRIBUTE_SEPARATOR is "." which is used for dotted attribute access like "article.title". An empty string after the dot makes no semantic sense in Django's template system.

## Relevant Context

This bug exists in Django 5.2.6 and likely affects earlier versions. The issue stems from a classic exception handling mistake where an exception meant to reject invalid input is accidentally caught by a broader exception handler meant for a different code path.

The Variable class is a fundamental component of Django's template system, responsible for parsing and resolving variable expressions in templates. While inputs like "2." are unlikely in production templates, the bug demonstrates a logic error that creates internally inconsistent Variable objects.

Django template documentation: https://docs.djangoproject.com/en/5.2/topics/templates/
Source code: django/template/base.py lines 822-848

## Proposed Fix

The fix requires ensuring the ValueError for trailing dots isn't caught by the outer exception handler. This can be achieved by either checking for trailing dots before the float conversion or using a more specific exception:

```diff
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -820,10 +820,10 @@ class Variable:
             # Try to interpret values containing a period or an 'e'/'E'
             # (possibly scientific notation) as a float;  otherwise, try int.
             if "." in var or "e" in var.lower():
-                self.literal = float(var)
                 # "2." is invalid
                 if var[-1] == ".":
                     raise TemplateSyntaxError("Invalid numeric literal: '%s'" % var)
+                self.literal = float(var)
             else:
                 self.literal = int(var)
         except ValueError:
```