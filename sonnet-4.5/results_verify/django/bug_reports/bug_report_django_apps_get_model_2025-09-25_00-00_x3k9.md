# Bug Report: django.apps.registry.Apps.get_model Unclear Error Message

**Target**: `django.apps.registry.Apps.get_model`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Apps.get_model()` method promises to "raise ValueError if called with a single argument that doesn't contain exactly one dot" according to its docstring. However, when called with a string containing multiple dots, it raises a confusing ValueError with message "too many values to unpack (expected 2)" instead of a clear error explaining the dot requirement.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.apps.registry import Apps


@given(st.lists(st.text(alphabet="abc", min_size=1, max_size=5), min_size=3, max_size=5))
def test_get_model_error_message_clarity(parts):
    model_string = ".".join(parts)

    registry = Apps(installed_apps=[])

    try:
        registry.get_model(model_string, model_name=None, require_ready=False)
    except ValueError as e:
        error_msg = str(e)
        assert "dot" in error_msg or "." in error_msg, (
            f"Error message '{error_msg}' doesn't explain the dot requirement. "
            f"Expected mention of 'dot' or '.' but got generic unpacking error."
        )
```

**Failing input**: `parts=['A', 'A', 'A']` (results in string `"A.A.A"`)

## Reproducing the Bug

```python
from django.apps.registry import Apps

registry = Apps(installed_apps=[])

try:
    registry.get_model("A.A.A", model_name=None, require_ready=False)
except ValueError as e:
    print(f"Actual error: {e}")
    print("Expected error: something like 'app_label must contain exactly one dot'")
```

Output:
```
Actual error: too many values to unpack (expected 2)
Expected error: something like 'app_label must contain exactly one dot'
```

## Why This Is A Bug

The docstring explicitly promises:
> Raise ValueError if called with a single argument that doesn't contain exactly one dot.

But the error message "too many values to unpack (expected 2)" exposes an implementation detail (the use of unpacking) and doesn't help the user understand what they did wrong. This violates the API contract and provides poor developer experience.

The same issue occurs with zero dots:
```python
registry.get_model("nodots", model_name=None, require_ready=False)
```
raises "ValueError: not enough values to unpack (expected 2, got 1)"

## Fix

```diff
--- a/django/apps/registry.py
+++ b/django/apps/registry.py
@@ -203,7 +203,11 @@ class Apps:
             self.check_apps_ready()

         if model_name is None:
-            app_label, model_name = app_label.split(".")
+            parts = app_label.split(".")
+            if len(parts) != 2:
+                raise ValueError(
+                    f"app_label argument must be in the form <app_label>.<model_name>, got '{app_label}'"
+                )
+            app_label, model_name = parts

         app_config = self.get_app_config(app_label)
```