# Bug Report: django.apps.get_model() Cryptic Error Message

**Target**: `django.apps.registry.Apps.get_model()`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When `Apps.get_model()` is called with a single argument that doesn't contain exactly one dot, it raises a ValueError with a cryptic Python unpacking error message instead of a clear, user-friendly message explaining the required format.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pytest

@given(st.text(min_size=1, max_size=50).filter(lambda x: '.' not in x))
@settings(max_examples=100)
def test_get_model_no_dot_clear_error(app_label_no_dot):
    with pytest.raises(ValueError) as exc_info:
        apps.get_model(app_label_no_dot)

    error_msg = str(exc_info.value)
    assert "exactly one dot" in error_msg or "format" in error_msg, \
        f"Error message should explain the format requirement, got: {error_msg}"
```

**Failing input**: `"contenttypes"` (any string without a dot)

## Reproducing the Bug

```python
import django
from django.conf import settings
from django.apps import apps

if not settings.configured:
    settings.configure(
        INSTALLED_APPS=['django.contrib.contenttypes'],
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        SECRET_KEY='test',
    )
django.setup()

try:
    apps.get_model("contenttypes")
except ValueError as e:
    print(f"Actual error: {e}")

try:
    apps.get_model("content.types.model")
except ValueError as e:
    print(f"Actual error: {e}")
```

Output:
```
Actual error: not enough values to unpack (expected 2, got 1)
Actual error: too many values to unpack (expected 2)
```

## Why This Is A Bug

The docstring at `django/apps/registry.py:196-198` explicitly states:

> Raise ValueError if called with a single argument that doesn't contain exactly one dot.

While a `ValueError` is correctly raised, the error message is a Python implementation detail ("not enough values to unpack") that doesn't explain the actual requirement to users. This violates the API contract by providing an unhelpful error message.

Expected behavior: A clear message like:
- `"app_label must be in the format 'app_label.model_name' with exactly one dot when model_name is not provided. Got: 'contenttypes'"`

Actual behavior:
- `"not enough values to unpack (expected 2, got 1)"`

## Fix

```diff
diff --git a/django/apps/registry.py b/django/apps/registry.py
index 1234567..abcdefg 100644
--- a/django/apps/registry.py
+++ b/django/apps/registry.py
@@ -203,7 +203,13 @@ class Apps:
             self.check_apps_ready()

         if model_name is None:
-            app_label, model_name = app_label.split(".")
+            parts = app_label.split(".")
+            if len(parts) != 2:
+                raise ValueError(
+                    "app_label must be in the format 'app_label.model_name' "
+                    f"with exactly one dot when model_name is not provided. "
+                    f"Got: {app_label!r}"
+                )
+            app_label, model_name = parts

         app_config = self.get_app_config(app_label)