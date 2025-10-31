# Bug Report: django.core.management.utils.handle_extensions Empty String Handling

**Target**: `django.core.management.utils.handle_extensions`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `handle_extensions()` function does not filter out empty strings that result from comma-separated extension lists, causing it to return `'.'` as a valid file extension when given inputs with consecutive commas, trailing commas, or comma-separated spaces.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.management.utils import handle_extensions

@given(st.lists(st.text(alphabet="abcdefghijklmnopqrstuvwxyz,. ", min_size=1, max_size=30)))
def test_handle_extensions_no_dot_only_extension(extensions):
    """
    Property: handle_extensions should never return '.' as an extension
    A lone dot is not a valid file extension.
    """
    result = handle_extensions(extensions)
    assert '.' not in result, f"Result contains invalid extension '.': {result}"
```

**Failing input**: `['py,,js']` (double comma creates empty string)

## Reproducing the Bug

```python
from django.core.management.utils import handle_extensions

result1 = handle_extensions(['py,,js'])
print(result1)


result2 = handle_extensions(['py,'])
print(result2)


result3 = handle_extensions(['py, ,js'])
print(result3)
```

**Output:**
```
{'.py', '.', '.js'}
{'.py', '.'}
{'.py', '.', '.js'}
```

## Why This Is A Bug

1. A file extension of just `'.'` is not semantically valid
2. The docstring examples show clean inputs without empty strings, suggesting this edge case wasn't considered
3. This could cause unexpected behavior in commands that use this function (e.g., `makemessages`) if they try to match files with extension `'.'`
4. The function is meant to parse user input, which may contain typos like trailing commas or double commas

## Fix

Filter out empty strings before adding the dot prefix:

```diff
--- a/django/core/management/utils.py
+++ b/django/core/management/utils.py
@@ -46,8 +46,9 @@ def handle_extensions(extensions):
     """
     ext_list = []
     for ext in extensions:
         ext_list.extend(ext.replace(" ", "").split(","))
     for i, ext in enumerate(ext_list):
-        if not ext.startswith("."):
+        if ext and not ext.startswith("."):
             ext_list[i] = ".%s" % ext_list[i]
+    ext_list = [ext for ext in ext_list if ext]
     return set(ext_list)
```

Alternatively, filter during the split phase:

```diff
--- a/django/core/management/utils.py
+++ b/django/core/management/utils.py
@@ -46,7 +46,7 @@ def handle_extensions(extensions):
     """
     ext_list = []
     for ext in extensions:
-        ext_list.extend(ext.replace(" ", "").split(","))
+        ext_list.extend([e for e in ext.replace(" ", "").split(",") if e])
     for i, ext in enumerate(ext_list):
         if not ext.startswith("."):
             ext_list[i] = ".%s" % ext_list[i]
     return set(ext_list)
```