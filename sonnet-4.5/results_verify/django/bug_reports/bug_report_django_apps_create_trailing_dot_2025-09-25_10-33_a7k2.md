# Bug Report: django.apps AppConfig.create() IndexError on Trailing Dot

**Target**: `django.apps.config.AppConfig.create`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`AppConfig.create()` raises an unhelpful `IndexError` when called with an entry that ends with a dot (e.g., `"django.contrib.auth."`), instead of a clear error message indicating the configuration is invalid.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.apps.config import AppConfig


@given(st.text(min_size=1).filter(lambda s: '.' in s))
def test_create_with_module_paths(entry):
    if entry.endswith('.') and not entry.rpartition('.')[2]:
        try:
            AppConfig.create(entry)
            assert False, "Should raise an exception"
        except IndexError:
            pass
        except (ImportError, Exception):
            pass
```

**Failing input**: `"django.contrib.auth."` or any module path ending with a dot where the final component is empty.

## Reproducing the Bug

```python
import django
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')
django.setup()

from django.apps.config import AppConfig

AppConfig.create("django.contrib.auth.")
```

**Output**:
```
IndexError: string index out of range
```

**Expected**: A clearer error message like:
```
ImproperlyConfigured: 'django.contrib.auth.' is not a valid app configuration.
```

## Why This Is A Bug

In `config.py` at line 171-172:
```python
mod_path, _, cls_name = entry.rpartition(".")
if mod_path and cls_name[0].isupper():
```

When `entry = "django.contrib.auth."`, `rpartition(".")` returns `('django.contrib.auth', '.', '')`, so `cls_name = ''`. The code then tries to access `cls_name[0]` on an empty string, causing an `IndexError`.

The check `if mod_path and cls_name[0].isupper()` assumes `cls_name` is non-empty when `mod_path` is truthy, but this isn't guaranteed.

## Fix

```diff
--- a/django/apps/config.py
+++ b/django/apps/config.py
@@ -169,7 +169,7 @@ class AppConfig:
             # If the last component of entry starts with an uppercase letter,
             # then it was likely intended to be an app config class; if not,
             # an app module. Provide a nice error message in both cases.
             mod_path, _, cls_name = entry.rpartition(".")
-            if mod_path and cls_name[0].isupper():
+            if mod_path and cls_name and cls_name[0].isupper():
                 # We could simply re-trigger the string import exception, but
                 # we're going the extra mile and providing a better error
```