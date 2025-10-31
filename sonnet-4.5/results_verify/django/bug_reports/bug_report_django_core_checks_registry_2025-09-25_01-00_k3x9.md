# Bug Report: django.core.checks.registry.CheckRegistry String Return Bug

**Target**: `django.core.checks.registry.CheckRegistry.run_checks()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a registered check function returns a string instead of a list of CheckMessage objects, `CheckRegistry.run_checks()` silently accepts it and iterates over the string character-by-character, resulting in a list of individual characters being returned instead of CheckMessage objects.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.checks.registry import CheckRegistry


@given(st.text(min_size=1))
def test_registry_check_returns_string_bug(error_string):
    registry = CheckRegistry()

    def bad_check(app_configs, **kwargs):
        return error_string

    registry.register(bad_check, "test")

    result = registry.run_checks()

    assert len(result) == len(error_string)
    assert result == list(error_string)
```

**Failing input**: Any string, e.g., `"error message"`

## Reproducing the Bug

```python
from django.core.checks.registry import CheckRegistry


registry = CheckRegistry()


def bad_check(app_configs, **kwargs):
    return "error message"


registry.register(bad_check, "test")

result = registry.run_checks()

print(f"Result: {result}")
print(f"Expected: list of CheckMessage objects")
print(f"Actual: {result}")
```

Output:
```
Result: ['e', 'r', 'r', 'o', 'r', ' ', 'm', 'e', 's', 's', 'a', 'g', 'e']
Expected: list of CheckMessage objects
Actual: ['e', 'r', 'r', 'o', 'r', ' ', 'm', 'e', 's', 's', 'a', 'g', 'e']
```

## Why This Is A Bug

The code in `CheckRegistry.run_checks()` verifies that check functions return an iterable:

```python
if not isinstance(new_errors, Iterable):
    raise TypeError(
        "The function %r did not return a list. All functions "
        "registered with the checks registry must return a list." % check,
    )
```

However, strings are iterable in Python, so this check passes when a string is returned. The code then calls `errors.extend(new_errors)`, which iterates over the string character-by-character, adding each character to the errors list. This violates the documented contract that check functions must return a list of CheckMessage objects.

This can lead to confusing errors downstream when code expects CheckMessage objects but receives string characters instead.

## Fix

```diff
--- a/django/core/checks/registry.py
+++ b/django/core/checks/registry.py
@@ -67,7 +67,7 @@ class CheckRegistry:

         for check in checks:
             new_errors = check(app_configs=app_configs, databases=databases)
-            if not isinstance(new_errors, Iterable):
+            if not isinstance(new_errors, (list, tuple)):
                 raise TypeError(
                     "The function %r did not return a list. All functions "
                     "registered with the checks registry must return a list." % check,
```

This fix changes the check from accepting any `Iterable` to only accepting `list` or `tuple`, which prevents strings from being accepted. Alternatively, the check could explicitly exclude strings:

```diff
--- a/django/core/checks/registry.py
+++ b/django/core/checks/registry.py
@@ -67,7 +67,7 @@ class CheckRegistry:

         for check in checks:
             new_errors = check(app_configs=app_configs, databases=databases)
-            if not isinstance(new_errors, Iterable):
+            if isinstance(new_errors, str) or not isinstance(new_errors, Iterable):
                 raise TypeError(
                     "The function %r did not return a list. All functions "
                     "registered with the checks registry must return a list." % check,
```