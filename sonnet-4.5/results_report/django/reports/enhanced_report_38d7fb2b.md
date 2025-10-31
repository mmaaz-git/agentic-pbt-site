# Bug Report: django.template.Variable Trailing Dot State Corruption

**Target**: `django.template.Variable.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Django template Variable class incorrectly sets both `literal` and `lookups` attributes when initialized with numeric strings ending in dots (e.g., "2.", "42."), violating the class invariant that these should be mutually exclusive. This causes `resolve()` to unexpectedly fail with VariableDoesNotExist.

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

# Run the test
if __name__ == '__main__':
    test_variable_trailing_dot_inconsistency()
```

<details>

<summary>
**Failing input**: `n=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 18, in <module>
    test_variable_trailing_dot_inconsistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 9, in test_variable_trailing_dot_inconsistency
    def test_variable_trailing_dot_inconsistency(n):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 14, in test_variable_trailing_dot_inconsistency
    assert False, f'Variable should not have both literal and lookups set'
           ^^^^^
AssertionError: Variable should not have both literal and lookups set
Falsifying example: test_variable_trailing_dot_inconsistency(
    n=0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import django.template

# Create a Variable with a trailing dot
v = django.template.Variable('2.')

# Inspect the state of the Variable
print(f'Variable created with input: "2."')
print(f'literal: {v.literal}')
print(f'lookups: {v.lookups}')
print('')

# Both literal and lookups are set - this violates the invariant
if v.literal is not None and v.lookups is not None:
    print('BUG: Both literal and lookups are set!')
    print('This violates the class invariant that a Variable should be either a literal OR a lookup.')
print('')

# Try to resolve the Variable
c = django.template.Context({})
print('Attempting to resolve the Variable...')
try:
    result = v.resolve(c)
    print(f'Result: {result}')
except django.template.VariableDoesNotExist as e:
    print(f'ERROR - VariableDoesNotExist: {e}')
```

<details>

<summary>
VariableDoesNotExist exception raised when resolving Variable('2.')
</summary>
```
Variable created with input: "2."
literal: 2.0
lookups: ('2', '')

BUG: Both literal and lookups are set!
This violates the class invariant that a Variable should be either a literal OR a lookup.

Attempting to resolve the Variable...
ERROR - VariableDoesNotExist: Failed lookup for key [2] in [{'True': True, 'False': False, 'None': None}, {}]
```
</details>

## Why This Is A Bug

This bug violates the fundamental design contract of the Variable class. The class is designed with a clear invariant: a Variable represents EITHER a literal value (stored in `self.literal`) OR a variable lookup path (stored in `self.lookups`), never both. This invariant is evident in the `resolve()` method implementation (lines 852-857 of base.py), which uses an if-else structure assuming mutual exclusivity.

The code explicitly recognizes that trailing dots are invalid, as shown by the comment on line 824: `# "2." is invalid`. The code attempts to reject such input by raising a ValueError on line 826. However, due to improper exception handling, this ValueError is caught by the outer except block (line 829), which then proceeds to set `self.lookups = tuple(var.split(VARIABLE_ATTRIBUTE_SEPARATOR))` on line 848.

This results in a corrupted Variable object where:
- `self.literal = 2.0` (from line 823, before the ValueError)
- `self.lookups = ('2', '')` (from line 848, after catching the ValueError)

The resolve() method then prioritizes lookups over literals (checking `if self.lookups is not None` first), causing it to attempt variable resolution and fail with a confusing VariableDoesNotExist error, even though a literal value exists.

## Relevant Context

This bug affects any Django template that processes numeric strings with trailing dots. Such strings can occur from:
- User input in forms
- Data formatting from external APIs
- CSV/Excel imports where numbers are formatted with trailing decimal points
- Database values that are string-formatted

The Django source code location: `/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py`

Key lines:
- Line 824: Comment stating "# '2.' is invalid"
- Line 823: `self.literal = float(var)` sets the literal
- Line 826: `raise ValueError` attempts to reject the input
- Line 829: `except ValueError:` catches the exception
- Line 848: `self.lookups = tuple(var.split(VARIABLE_ATTRIBUTE_SEPARATOR))` sets lookups
- Lines 852-857: resolve() method assumes mutual exclusivity

## Proposed Fix

```diff
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -822,6 +822,7 @@ class Variable:
             if "." in var or "e" in var.lower():
                 self.literal = float(var)
                 # "2." is invalid
                 if var[-1] == ".":
+                    self.literal = None
                     raise ValueError
             else:
```