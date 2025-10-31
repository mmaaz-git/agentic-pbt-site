# Bug Report: django.core.checks.registry.CheckRegistry.run_checks String Iteration Bug

**Target**: `django.core.checks.registry.CheckRegistry.run_checks`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When a check function returns a string instead of a list, `CheckRegistry.run_checks()` silently iterates over the string characters and adds each one to the errors list, violating the documented contract that check functions must return a list.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

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

# Run the test
if __name__ == "__main__":
    test_run_checks_should_reject_string_return_value()
```

<details>

<summary>
**Failing input**: `error_string='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 28, in <module>
    test_run_checks_should_reject_string_return_value()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 10, in test_run_checks_should_reject_string_return_value
    def test_run_checks_should_reject_string_return_value(error_string):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 20, in test_run_checks_should_reject_string_return_value
    assert isinstance(err, CheckMessage), (
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
AssertionError: Expected CheckMessage, got str: '0'. This suggests the check function returned a string which was incorrectly treated as an iterable of CheckMessages.
Falsifying example: test_run_checks_should_reject_string_return_value(
    error_string='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.core.checks.registry import CheckRegistry

# Create a registry instance
registry = CheckRegistry()

# Define a buggy check function that returns a string instead of a list
def buggy_check(app_configs=None, **kwargs):
    return "error message"

# Register the buggy check
registry.register(buggy_check)

# Run the checks
errors = registry.run_checks()

# Display the errors
print(f"Errors list: {errors}")
print(f"Errors type: {type(errors)}")
print(f"Number of items in errors: {len(errors)}")
print(f"First item: {repr(errors[0]) if errors else 'No errors'}")
print(f"All items (repr): {[repr(e) for e in errors]}")
```

<details>

<summary>
String gets split into individual characters
</summary>
```
Errors list: ['e', 'r', 'r', 'o', 'r', ' ', 'm', 'e', 's', 's', 'a', 'g', 'e']
Errors type: <class 'list'>
Number of items in errors: 13
First item: 'e'
All items (repr): ["'e'", "'r'", "'r'", "'o'", "'r'", "' '", "'m'", "'e'", "'s'", "'s'", "'a'", "'g'", "'e'"]
```
</details>

## Why This Is A Bug

This violates the documented contract in multiple ways:

1. **Documentation Contract Violation**: The `register()` method docstring (line 36-38) explicitly states: "The function should receive **kwargs and return list of Errors and Warnings." The error message at line 92-93 also states: "All functions registered with the checks registry must return a list."

2. **Inconsistent Validation**: While the documentation and error message require a "list", the actual validation at line 90 only checks `isinstance(new_errors, Iterable)`, which accepts strings since strings are iterables in Python.

3. **Silent Data Corruption**: Instead of raising an error, the code treats the string as an iterable and adds individual characters ('e', 'r', 'r', 'o', 'r', etc.) to the errors list. This behavior is clearly unintended and nonsensical.

4. **Type Safety Violation**: The errors list is expected to contain CheckMessage objects (or subclasses like Error/Warning), but ends up containing individual string characters when this bug occurs.

5. **Misleading Error Behavior**: The error message claims to check for a "list" but the actual check allows any iterable, creating confusion for developers who may rely on the error message for guidance.

## Relevant Context

The issue stems from Python's design where strings are iterables. The code at line 95 uses `errors.extend(new_errors)`, which when given a string, iterates over each character. This is a common Python gotcha that Django's check framework should guard against.

All of Django's built-in check functions correctly return lists (empty lists `[]` or lists of CheckMessage objects), never strings. This shows the intended API design. The bug likely went unnoticed because Django's own code follows the documented contract correctly.

Documentation references:
- Source code: `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/checks/registry.py`
- Lines 36-38: Register method docstring
- Lines 79-81: Run_checks method docstring
- Lines 90-95: The problematic validation logic

## Proposed Fix

```diff
--- a/django/core/checks/registry.py
+++ b/django/core/checks/registry.py
@@ -88,6 +88,11 @@ class CheckRegistry:
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