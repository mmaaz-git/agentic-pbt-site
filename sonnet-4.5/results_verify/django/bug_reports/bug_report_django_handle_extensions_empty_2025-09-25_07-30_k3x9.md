# Bug Report: django.core.management.utils.handle_extensions Invalid Extension

**Target**: `django.core.management.utils.handle_extensions`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `handle_extensions` function produces an invalid file extension '.' (single dot) when processing comma-separated extension lists that contain empty strings, such as those created by double commas, leading commas, or trailing commas in user input.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.management.utils import handle_extensions

@given(st.lists(st.text(alphabet=' ,', min_size=1, max_size=20), min_size=1, max_size=5))
def test_handle_extensions_no_single_dot(separator_strings):
    result = handle_extensions(separator_strings)
    assert '.' not in result, f"Invalid extension '.' should not be in result"
```

**Failing input**: `[',']` or `['html,,css']` or `['']`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
from django.core.management.utils import handle_extensions

print("Test case 1: Empty string")
result = handle_extensions([''])
print(f"Result: {result}")
assert '.' in result

print("\nTest case 2: Double comma")
result = handle_extensions(['html,,css'])
print(f"Result: {result}")
assert '.' in result

print("\nTest case 3: Trailing comma")
result = handle_extensions(['html,'])
print(f"Result: {result}")
assert '.' in result

print("\nTest case 4: Leading comma")
result = handle_extensions([',html'])
print(f"Result: {result}")
assert '.' in result
```

## Why This Is A Bug

The `handle_extensions` function is documented to "organize multiple extensions that are separated with commas" and return valid file extensions. A single dot '.' is not a valid file extension for any file system.

This bug can occur in real usage because:
1. The `makemessages` command documents that users should "Separate multiple extensions with commas" via the `--extension` flag
2. Users could accidentally type `--extension "html,,css"` (double comma) or `--extension "html,"` (trailing comma)
3. The resulting '.' extension would match no files (or potentially cause errors in file matching logic)

The root cause is that when splitting by comma, empty strings can be created (e.g., `'html,,css'.split(',')` produces `['html', '', 'css']`), and the code unconditionally adds a dot prefix to all items including empty strings, producing '.'.

## Fix

```diff
diff --git a/django/core/management/utils.py b/django/core/management/utils.py
index 1234567..abcdefg 100644
--- a/django/core/management/utils.py
+++ b/django/core/management/utils.py
@@ -46,8 +46,10 @@ def handle_extensions(extensions):
     """
     ext_list = []
     for ext in extensions:
         ext_list.extend(ext.replace(" ", "").split(","))
     for i, ext in enumerate(ext_list):
-        if not ext.startswith("."):
+        if ext and not ext.startswith("."):
             ext_list[i] = ".%s" % ext_list[i]
+        elif not ext:
+            ext_list[i] = None
-    return set(ext_list)
+    return {e for e in ext_list if e and e != '.'}
```