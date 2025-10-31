# Bug Report: django.template.Variable Trailing Dot Inconsistency

**Target**: `django.template.Variable`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Variable` class incorrectly handles numeric strings with trailing dots (e.g., "2.", "42."). It sets both `literal` and `lookups` attributes, violating the invariant that a Variable should represent either a literal value OR a variable lookup, not both. This causes `resolve()` to fail with a VariableDoesNotExist exception.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django.template
from hypothesis import given, strategies as st


@given(st.integers(min_value=0, max_value=1000))
def test_variable_trailing_dot_inconsistency(n):
    var_string = f'{n}.'
    v = django.template.Variable(var_string)

    if v.literal is not None and v.lookups is not None:
        assert False, f'Variable should not have both literal and lookups set'
```

**Failing input**: `n=0` (or any integer)

## Reproducing the Bug

```python
import django.template

v = django.template.Variable('2.')

print(f'literal: {v.literal}')
print(f'lookups: {v.lookups}')

c = django.template.Context({})
try:
    result = v.resolve(c)
    print(f'Result: {result}')
except django.template.VariableDoesNotExist as e:
    print(f'ERROR: {e}')
```

**Output:**
```
literal: 2.0
lookups: ('2', '')
ERROR: Failed lookup for key [2] in [{'True': True, 'False': False, 'None': None}, {}]
```

## Why This Is A Bug

The code in `Variable.__init__` attempts to reject trailing dots as invalid:

```python
if "." in var or "e" in var.lower():
    self.literal = float(var)
    # "2." is invalid
    if var[-1] == ".":
        raise ValueError
```

However, the `ValueError` is caught by the outer try-except block, which then proceeds to set `self.lookups`. This results in a Variable with both `literal=2.0` and `lookups=('2', '')`, violating the class's invariant that a variable is either a literal or a lookup.

This causes `resolve()` to fail because it attempts to perform a lookup even though `literal` is set, leading to unexpected VariableDoesNotExist errors.

## Fix

The fix is to reset `self.literal = None` when re-raising the ValueError for trailing dots, or to use a different exception type that won't be caught by the outer handler:

```diff
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -18,6 +18,7 @@ class Variable:
             if "." in var or "e" in var.lower():
                 self.literal = float(var)
                 # "2." is invalid
                 if var[-1] == ".":
+                    self.literal = None
                     raise ValueError
             else:
```

Alternatively, the code should use a more specific exception or check this condition before attempting to parse as a float.