# Bug Report: django.template.Variable Trailing Dot State Corruption

**Target**: `django.template.Variable`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When Variable is initialized with a numeric string ending in a dot (e.g., '42.'), both `literal` and `lookups` attributes become set simultaneously, violating the class invariant and causing resolution to fail despite successful Variable creation.

## Property-Based Test

```python
#!/usr/bin/env python
"""Property-based test for django.template.Variable trailing dot bug"""

from django.template import Variable, Context
from hypothesis import given, settings, strategies as st, example


@settings(max_examples=100)
@example(n=42)
@example(n=0)
@example(n=123)
@given(st.integers())
def test_variable_numeric_string_with_trailing_dot_should_be_resolvable(n):
    s = str(n) + '.'
    var = Variable(s)

    if var.literal is not None:
        ctx = Context({})
        resolved = var.resolve(ctx)
        assert resolved == var.literal


if __name__ == "__main__":
    test_variable_numeric_string_with_trailing_dot_should_be_resolvable()
```

<details>

<summary>
**Failing input**: `n=42` (and any integer)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 24, in <module>
  |     test_variable_numeric_string_with_trailing_dot_should_be_resolvable()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 9, in test_variable_numeric_string_with_trailing_dot_should_be_resolvable
  |     @example(n=42)
  |
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures in explicit examples. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 891, in _resolve_lookup
    |     current = current[bit]
    |               ~~~~~~~^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/context.py", line 88, in __getitem__
    |     raise KeyError(key)
    | KeyError: '42'
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 897, in _resolve_lookup
    |     if isinstance(current, BaseContext) and getattr(
    |                                             ~~~~~~~^
    |         type(current), bit
    |         ^^^^^^^^^^^^^^^^^^
    |     ):
    |     ^
    | AttributeError: type object 'Context' has no attribute '42'
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 907, in _resolve_lookup
    |     current = current[int(bit)]
    |               ~~~~~~~^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/context.py", line 88, in __getitem__
    |     raise KeyError(key)
    | KeyError: 42
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 19, in test_variable_numeric_string_with_trailing_dot_should_be_resolvable
    |     resolved = var.resolve(ctx)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 854, in resolve
    |     value = self._resolve_lookup(context)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 914, in _resolve_lookup
    |     raise VariableDoesNotExist(
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |     ...<2 lines>...
    |     )  # missing attribute
    |     ^
    | django.template.base.VariableDoesNotExist: Failed lookup for key [42] in [{'True': True, 'False': False, 'None': None}, {}]
    | Falsifying explicit example: test_variable_numeric_string_with_trailing_dot_should_be_resolvable(
    |     n=42,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 891, in _resolve_lookup
    |     current = current[bit]
    |               ~~~~~~~^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/context.py", line 88, in __getitem__
    |     raise KeyError(key)
    | KeyError: '0'
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 897, in _resolve_lookup
    |     if isinstance(current, BaseContext) and getattr(
    |                                             ~~~~~~~^
    |         type(current), bit
    |         ^^^^^^^^^^^^^^^^^^
    |     ):
    |     ^
    | AttributeError: type object 'Context' has no attribute '0'
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 907, in _resolve_lookup
    |     current = current[int(bit)]
    |               ~~~~~~~^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/context.py", line 88, in __getitem__
    |     raise KeyError(key)
    | KeyError: 0
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 19, in test_variable_numeric_string_with_trailing_dot_should_be_resolvable
    |     resolved = var.resolve(ctx)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 854, in resolve
    |     value = self._resolve_lookup(context)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 914, in _resolve_lookup
    |     raise VariableDoesNotExist(
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |     ...<2 lines>...
    |     )  # missing attribute
    |     ^
    | django.template.base.VariableDoesNotExist: Failed lookup for key [0] in [{'True': True, 'False': False, 'None': None}, {}]
    | Falsifying explicit example: test_variable_numeric_string_with_trailing_dot_should_be_resolvable(
    |     n=0,
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 891, in _resolve_lookup
    |     current = current[bit]
    |               ~~~~~~~^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/context.py", line 88, in __getitem__
    |     raise KeyError(key)
    | KeyError: '123'
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 897, in _resolve_lookup
    |     if isinstance(current, BaseContext) and getattr(
    |                                             ~~~~~~~^
    |         type(current), bit
    |         ^^^^^^^^^^^^^^^^^^
    |     ):
    |     ^
    | AttributeError: type object 'Context' has no attribute '123'
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 907, in _resolve_lookup
    |     current = current[int(bit)]
    |               ~~~~~~~^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/context.py", line 88, in __getitem__
    |     raise KeyError(key)
    | KeyError: 123
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 19, in test_variable_numeric_string_with_trailing_dot_should_be_resolvable
    |     resolved = var.resolve(ctx)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 854, in resolve
    |     value = self._resolve_lookup(context)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/base.py", line 914, in _resolve_lookup
    |     raise VariableDoesNotExist(
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |     ...<2 lines>...
    |     )  # missing attribute
    |     ^
    | django.template.base.VariableDoesNotExist: Failed lookup for key [123] in [{'True': True, 'False': False, 'None': None}, {}]
    | Falsifying explicit example: test_variable_numeric_string_with_trailing_dot_should_be_resolvable(
    |     n=123,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python
"""Minimal reproduction of the django.template.Variable trailing dot bug"""

from django.template import Variable, Context

# Create a Variable with a numeric string ending in a dot
var = Variable('42.')

# Show the inconsistent internal state
print(f"Variable created with '42.':")
print(f"  literal: {var.literal}")
print(f"  lookups: {var.lookups}")
print()

# Try to resolve it
ctx = Context({})
try:
    result = var.resolve(ctx)
    print(f"Resolution succeeded: {result}")
except Exception as e:
    print(f"Resolution failed with: {type(e).__name__}: {e}")
```

<details>

<summary>
VariableDoesNotExist exception on resolution
</summary>
```
Variable created with '42.':
  literal: 42.0
  lookups: ('42', '')

Resolution failed with: VariableDoesNotExist: Failed lookup for key [42] in [{'True': True, 'False': False, 'None': None}, {}]
```
</details>

## Why This Is A Bug

The Variable class design assumes that `literal` and `lookups` are mutually exclusive - only one should be set. The `resolve()` method at line 852 of `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/template/base.py` confirms this with its if-else structure:

```python
if self.lookups is not None:
    value = self._resolve_lookup(context)
else:
    value = self.literal
```

When processing '42.', the code execution flow causes both fields to be set:

1. Line 823: `float('42.')` succeeds, setting `self.literal = 42.0`
2. Line 825-826: The code explicitly checks `if var[-1] == "."` and raises ValueError with comment "# '2.' is invalid"
3. Line 829: The ValueError handler begins, intending to process this as a variable lookup instead
4. Line 848: Sets `self.lookups = ('42', '')` splitting on the dot separator
5. **Bug**: The handler never clears `self.literal`, leaving both fields populated

This violates the invariant that either `literal` OR `lookups` should be set. Since `resolve()` checks `lookups` first, it attempts variable resolution which fails because '42' doesn't exist in the context and the empty string after the dot is invalid for lookups.

## Relevant Context

The code at line 824-826 shows clear intent to reject numeric strings ending with dots:
```python
# "2." is invalid
if var[-1] == ".":
    raise ValueError
```

However, the exception handling fails to maintain consistency. The comment at line 830 states "# A ValueError means that the variable isn't a number" but this isn't accurate - the value WAS successfully parsed as a number (42.0) before being rejected for the trailing dot.

This bug affects any numeric string ending with a dot:
- '42.' becomes literal=42.0, lookups=('42', '')
- '0.' becomes literal=0.0, lookups=('0', '')
- '123.456.' becomes literal=123.456, lookups=('123.456', '')

Normal cases work correctly:
- '42' → literal=42, lookups=None
- '42.0' → literal=42.0, lookups=None
- '.42' → literal=0.42, lookups=None

## Proposed Fix

```diff
--- a/django/template/base.py
+++ b/django/template/base.py
@@ -827,6 +827,7 @@ class Variable:
             else:
                 self.literal = int(var)
         except ValueError:
+            self.literal = None
             # A ValueError means that the variable isn't a number.
             if var[0:2] == "_(" and var[-1] == ")":
                 # The result of the lookup should be translated at rendering
```