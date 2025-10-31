# Bug Report: django.template.Variable Trailing Period Handling

**Target**: `django.template.Variable`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Variable class attempts to reject numeric strings with trailing periods (e.g., "1.", "2.") but fails to do so correctly, leaving the Variable object in an inconsistent state that causes crashes when resolved.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.template import Variable

@given(st.integers(min_value=1, max_value=1000000))
def test_variable_trailing_period_invalid(n):
    var_str = f'{n}.'
    var = Variable(var_str)
    if var.literal is not None:
        assert False, f'Expected ValueError for {var_str}, but got literal: {var.literal}'
```

**Failing input**: `n=1` (produces string "1.")

## Reproducing the Bug

```python
from django.template import Variable, Context

var = Variable('1.')
print(f'literal: {var.literal}')
print(f'lookups: {var.lookups}')

context = Context({'1': 'test'})
result = var.resolve(context)
```

Output:
```
literal: 1.0
lookups: ('1', '')
Traceback (most recent call last):
  ...
django.template.base.VariableDoesNotExist: Failed lookup for key [] in 'test'
```

## Why This Is A Bug

The source code contains a comment stating "2." is invalid and includes a check to reject it:

```python
if "." in var or "e" in var.lower():
    self.literal = float(var)
    # "2." is invalid
    if var[-1] == ".":
        raise ValueError
```

However, the ValueError is raised AFTER `self.literal = float(var)` executes, so when the exception is caught by the outer try-except block, the Variable object is left in an inconsistent state with both `literal` and `lookups` set. This violates the invariant that a Variable should have either a literal value OR lookups, not both.

When such a Variable is resolved, it attempts to use the lookups (since `lookups is not None`), which includes an empty string, leading to a crash.

## Fix

Move the trailing period check before setting `self.literal`:

```diff
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -123,9 +123,10 @@ class Variable:
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