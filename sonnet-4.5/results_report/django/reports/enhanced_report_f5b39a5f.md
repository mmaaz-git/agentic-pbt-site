# Bug Report: django.views.i18n.JavaScriptCatalog.get_plural IndexError on Malformed Plural-Forms Headers

**Target**: `django.views.i18n.JavaScriptCatalog.get_plural`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_plural()` method in Django's JavaScriptCatalog crashes with an IndexError when processing translation catalogs that have malformed Plural-Forms headers missing the `plural=` expression part.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

from hypothesis import given, strategies as st, assume
from django.views.i18n import JavaScriptCatalog
from unittest.mock import Mock


@given(st.text(min_size=1))
def test_get_plural_never_crashes(plural_forms_value):
    assume("plural=" not in plural_forms_value or not any(
        part.strip().startswith("plural=")
        for part in plural_forms_value.split(";")
    ))

    catalog = JavaScriptCatalog()
    catalog.translation = Mock()
    catalog.translation._catalog = {
        "": f"Plural-Forms: {plural_forms_value}"
    }

    result = catalog.get_plural()


if __name__ == "__main__":
    test_get_plural_never_crashes()
```

<details>

<summary>
**Failing input**: `plural_forms_value='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 29, in <module>
    test_get_plural_never_crashes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 13, in test_get_plural_never_crashes
    def test_get_plural_never_crashes(plural_forms_value):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 25, in test_get_plural_never_crashes
    result = catalog.get_plural()
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/views/i18n.py", line 171, in get_plural
    plural = [
             ~
    ...<2 lines>...
        if el.strip().startswith("plural=")
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ][0].split("=", 1)[1]
    ~^^^
IndexError: list index out of range
Falsifying example: test_get_plural_never_crashes(
    plural_forms_value='0',
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

from django.views.i18n import JavaScriptCatalog
from unittest.mock import Mock

# Create a JavaScriptCatalog instance
catalog = JavaScriptCatalog()

# Mock the translation object with a malformed Plural-Forms header
# This header has nplurals but is missing the plural= part
catalog.translation = Mock()
catalog.translation._catalog = {
    "": "Plural-Forms: nplurals=2;"
}

# Try to get the plural - this should crash with IndexError
try:
    result = catalog.get_plural()
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
IndexError: list index out of range
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/repo.py", line 22, in <module>
    result = catalog.get_plural()
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/views/i18n.py", line 171, in get_plural
    plural = [
             ~
    ...<2 lines>...
        if el.strip().startswith("plural=")
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ][0].split("=", 1)[1]
    ~^^^
IndexError: list index out of range
IndexError: list index out of range
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Inconsistent error handling**: The related `_num_plurals` property (lines 143-151) gracefully handles malformed Plural-Forms headers by returning a default value of 2, but `get_plural()` crashes instead.

2. **Template expects None handling**: The JavaScript template (`i18n_catalog.js`, lines 7-18) explicitly checks `{% if plural %}` and provides a default fallback function when plural is None/falsy, indicating the system is designed to handle missing plural expressions gracefully.

3. **Common real-world scenario**: Translation files can be malformed, especially when coming from third-party sources, during development, or when partially generated. According to GNU gettext standards, while both `nplurals=` and `plural=` are typically included together, the standard doesn't mandate that systems must crash when `plural=` is missing.

4. **Breaks entire JavaScript i18n**: When this crash occurs, the entire JavaScript catalog view fails to load, breaking all JavaScript translations on the affected pages, not just plural forms.

## Relevant Context

The bug occurs in the `get_plural()` method at lines 165-176 of `/django/views/i18n.py`. The problematic code attempts to extract the plural expression by:
1. Splitting the Plural-Forms string by semicolons
2. Filtering for parts that start with "plural="
3. Accessing the first element `[0]` of the filtered list
4. Extracting the value after the equals sign

When no "plural=" part exists, the filtered list is empty, causing `[0]` to raise an IndexError.

The JavaScript template that uses this value already has robust fallback handling:
- Line 7: `{% if plural %}` checks if plural exists
- Line 17: Provides a default implementation when plural is None: `django.pluralidx = function(count) { return (count == 1) ? 0 : 1; };`

This indicates the system architecture expects `get_plural()` might return None and handles it appropriately.

## Proposed Fix

```diff
--- a/django/views/i18n.py
+++ b/django/views/i18n.py
@@ -165,11 +165,15 @@ class JavaScriptCatalog(View):
     def get_plural(self):
         plural = self._plural_string
         if plural is not None:
             # This should be a compiled function of a typical plural-form:
             # Plural-Forms: nplurals=3; plural=n%10==1 && n%100!=11 ? 0 :
             #               n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2;
-            plural = [
+            plural_parts = [
                 el.strip()
                 for el in plural.split(";")
                 if el.strip().startswith("plural=")
-            ][0].split("=", 1)[1]
+            ]
+            if plural_parts:
+                plural = plural_parts[0].split("=", 1)[1]
+            else:
+                plural = None
         return plural
```