# Bug Report: django.template.Variable Inconsistent State with Trailing Period Numeric Literals

**Target**: `django.template.Variable`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Variable class enters an inconsistent internal state when parsing numeric literals with trailing periods (e.g., "10."), setting both `literal` and `lookups` attributes simultaneously, which violates the class's design that these should be mutually exclusive.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import django.template

@given(st.integers(min_value=0, max_value=1000))
def test_integer_trailing_period_property(num):
    text = f"{num}."
    var = django.template.Variable(text)

    if var.literal is not None and var.lookups is not None:
        assert False, f"Both literal ({var.literal}) and lookups ({var.lookups}) are set for '{text}'"

if __name__ == "__main__":
    test_integer_trailing_period_property()
```

<details>

<summary>
**Failing input**: `num=0` (or any integer from 0 to 1000)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 13, in <module>
    test_integer_trailing_period_property()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 5, in test_integer_trailing_period_property
    def test_integer_trailing_period_property(num):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 10, in test_integer_trailing_period_property
    assert False, f"Both literal ({var.literal}) and lookups ({var.lookups}) are set for '{text}'"
           ^^^^^
AssertionError: Both literal (0.0) and lookups (('0', '')) are set for '0.'
Falsifying example: test_integer_trailing_period_property(
    num=0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from django.template import Variable

# Create a Variable with a number followed by trailing period
var = Variable("10.")

# Display the internal state - both literal and lookups are set
print(f"literal: {var.literal}")
print(f"lookups: {var.lookups}")

# This shows the inconsistent state - both are set when they should be mutually exclusive
assert var.literal == 10.0, f"Expected literal to be 10.0, got {var.literal}"
assert var.lookups == ('10', ''), f"Expected lookups to be ('10', ''), got {var.lookups}"

# Try to resolve the variable - this raises an exception instead of returning the literal
try:
    result = var.resolve({})
    print(f"Result: {result}")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
```

<details>

<summary>
VariableDoesNotExist exception raised instead of returning literal value
</summary>
```
literal: 10.0
lookups: ('10', '')
VariableDoesNotExist: Failed lookup for key [10] in {}
```
</details>

## Why This Is A Bug

The Variable class contains explicit code (line 824-826 in django/template/base.py) with a comment stating "# '2.' is invalid" and attempts to raise a ValueError for numeric literals ending with a period. However, due to a logic error in the implementation:

1. Line 823 successfully sets `self.literal = float("10.")` to 10.0 (since Python's float() accepts trailing periods)
2. Line 826 raises ValueError when detecting the trailing period
3. The ValueError is caught on line 829, but `self.literal` remains set from step 1
4. Execution continues through lines 830-848, eventually setting `self.lookups = ('10', '')`

This creates an invalid state where both `literal` and `lookups` are set, violating the mutual exclusivity assumption throughout the codebase. The `resolve()` method (line 852) checks `lookups` first, so it attempts variable resolution instead of returning the literal value, causing a confusing `VariableDoesNotExist` error.

## Relevant Context

The Variable class is an internal Django template system component that parses template variable expressions. Variables should be either:
- Literals (numbers or quoted strings) with `literal` set and `lookups=None`
- Variable lookups (e.g., "object.attribute") with `lookups` set and `literal=None`

The resolve() method implementation assumes this mutual exclusivity:
- Lines 852-854: If `lookups is not None`, perform variable resolution
- Lines 855-857: Otherwise, return the literal value

Django's template documentation doesn't explicitly address numeric literals with trailing periods, but the code comment clearly indicates the developers intended to reject them. This bug only affects direct instantiation of Variable objects or template expressions like `{{ 10. }}`, which are rare in practice.

## Proposed Fix

The issue can be fixed by checking for trailing periods before setting the literal value:

```diff
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -820,11 +820,11 @@ class Variable:
             # Try to interpret values containing a period or an 'e'/'E'
             # (possibly scientific notation) as a float;  otherwise, try int.
             if "." in var or "e" in var.lower():
-                self.literal = float(var)
                 # "2." is invalid
                 if var[-1] == ".":
                     raise ValueError
+                self.literal = float(var)
             else:
                 self.literal = int(var)
         except ValueError:
```