# Bug Report: django.core.checks CheckRegistry.run_checks String Iteration Bug

**Target**: `django.core.checks.registry.CheckRegistry.run_checks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a check function incorrectly returns a string instead of a list of CheckMessage objects, `CheckRegistry.run_checks()` silently treats the string as an iterable and adds each character to the errors list, instead of raising a TypeError.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st
from django.core.checks import CheckMessage
from django.core.checks.registry import CheckRegistry


@given(st.text(min_size=1))
def test_run_checks_should_reject_string_return_value(error_string):
    registry = CheckRegistry()

    def my_check(app_configs=None, **kwargs):
        return error_string

    registry.register(my_check)
    errors = registry.run_checks()

    for err in errors:
        assert isinstance(err, CheckMessage), (
            f"Expected CheckMessage, got {type(err).__name__}: {repr(err)}. "
            f"This suggests the check function returned a string which was "
            f"incorrectly treated as an iterable of CheckMessages."
        )
```

**Failing input**: `error_string='0'` (or any string)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.checks.registry import CheckRegistry

registry = CheckRegistry()

def buggy_check(app_configs=None, **kwargs):
    return "error message"

registry.register(buggy_check)
errors = registry.run_checks()

print(f"Errors list: {errors}")
```

**Output:**
```
Errors list: ['e', 'r', 'r', 'o', 'r', ' ', 'm', 'e', 's', 's', 'a', 'g', 'e']
```

## Why This Is A Bug

1. **Contract Violation**: The docstring and error message for `run_checks()` state that check functions "must return a list", but the validation only checks `isinstance(new_errors, Iterable)`, which accepts strings.

2. **Silent Data Corruption**: When a check function accidentally returns a string (e.g., due to a typo or logic error), the method silently corrupts the errors list by adding individual characters instead of raising a clear error.

3. **Misleading Error Message**: The TypeError message says "must return a list" but only validates that it's an Iterable, which is inconsistent.

4. **Type Safety**: The returned errors list should contain only CheckMessage objects, but can contain arbitrary strings when this bug occurs.

## Fix

The validation should explicitly check for strings and reject them, or the error message should be updated to match the actual validation. The recommended fix is to add a string check:

```diff
--- a/django/core/checks/registry.py
+++ b/django/core/checks/registry.py
@@ -88,6 +88,9 @@ class CheckRegistry:
         for check in checks:
             new_errors = check(app_configs=app_configs, databases=databases)
+            if isinstance(new_errors, str):
+                raise TypeError(
+                    "The function %r returned a string. All functions "
+                    "registered with the checks registry must return a list of "
+                    "CheckMessage objects, not a string." % check,
+                )
             if not isinstance(new_errors, Iterable):
                 raise TypeError(
                     "The function %r did not return a list. All functions "
```

Alternatively, update the error message to match the actual validation:

```diff
--- a/django/core/checks/registry.py
+++ b/django/core/checks/registry.py
@@ -90,7 +90,7 @@ class CheckRegistry:
             if not isinstance(new_errors, Iterable):
                 raise TypeError(
-                    "The function %r did not return a list. All functions "
-                    "registered with the checks registry must return a list." % check,
+                    "The function %r did not return an iterable. All functions "
+                    "registered with the checks registry must return an iterable of CheckMessage objects." % check,
                 )
```