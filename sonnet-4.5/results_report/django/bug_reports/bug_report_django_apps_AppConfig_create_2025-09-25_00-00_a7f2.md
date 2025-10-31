# Bug Report: django.apps.AppConfig.create IndexError on Trailing Dot

**Target**: `django.apps.AppConfig.create`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `AppConfig.create()` is called with an entry that ends with a dot (e.g., `"myapp."`), the method crashes with an `IndexError: string index out of range` instead of providing a helpful error message.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from django.apps.config import AppConfig


@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz.', min_size=1, max_size=30))
@settings(max_examples=1000)
def test_create_rpartition_edge_cases(entry):
    mod_path, _, cls_name = entry.rpartition(".")

    if mod_path and not cls_name:
        try:
            config = AppConfig.create(entry)
        except IndexError as e:
            assert False, f"IndexError should not occur: {e}"
        except Exception:
            pass
```

**Failing input**: `"myapp."`

## Reproducing the Bug

```python
from django.apps.config import AppConfig

entry = "myapp."
config = AppConfig.create(entry)
```

**Output:**
```
IndexError: string index out of range
  File "django/apps/config.py", line 172, in create
    if mod_path and cls_name[0].isupper():
                    ^^^^^^^^^^^
```

## Why This Is A Bug

The code at line 172 in `config.py` attempts to check if the first character of `cls_name` is uppercase without first verifying that `cls_name` is non-empty. When an entry ends with a dot, `entry.rpartition(".")` produces an empty `cls_name`, causing an IndexError when accessing `cls_name[0]`.

This violates the method's error handling contract - configuration errors should produce helpful error messages (ImportError or ImproperlyConfigured), not IndexError.

## Fix

```diff
--- a/django/apps/config.py
+++ b/django/apps/config.py
@@ -169,7 +169,7 @@ class AppConfig:
             # then it was likely intended to be an app config class; if not,
             # an app module. Provide a nice error message in both cases.
             mod_path, _, cls_name = entry.rpartition(".")
-            if mod_path and cls_name[0].isupper():
+            if mod_path and cls_name and cls_name[0].isupper():
                 # We could simply re-trigger the string import exception, but
                 # we're going the extra mile and providing a better error
                 # message for typos in INSTALLED_APPS.
```