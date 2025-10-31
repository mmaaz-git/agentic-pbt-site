# Bug Report: django.views.i18n.JavaScriptCatalog.get_plural IndexError on Malformed Plural-Forms

**Target**: `django.views.i18n.JavaScriptCatalog.get_plural`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_plural()` method in `JavaScriptCatalog` crashes with an `IndexError` when the Plural-Forms header in a translation catalog is malformed (specifically, when it's missing the `plural=` part).

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
```

**Failing input**: `plural_forms_value='0'` (or any string without "plural=" after semicolon split)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

from django.views.i18n import JavaScriptCatalog
from unittest.mock import Mock

catalog = JavaScriptCatalog()
catalog.translation = Mock()
catalog.translation._catalog = {
    "": "Plural-Forms: nplurals=2;"
}

catalog.get_plural()
```

Output:
```
IndexError: list index out of range
```

## Why This Is A Bug

The `get_plural()` method assumes that the Plural-Forms header will always contain a `plural=` expression. When it doesn't, the list comprehension at line 171-175 produces an empty list, and accessing `[0]` raises an `IndexError`.

This is inconsistent with the `_num_plurals` property, which gracefully falls back to a default value of 2 when the header is malformed. The `get_plural()` method should similarly handle malformed input gracefully instead of crashing.

## Fix

```diff
--- a/django/views/i18n.py
+++ b/django/views/i18n.py
@@ -165,11 +165,13 @@ class JavaScriptCatalog(View):
     def get_plural(self):
         plural = self._plural_string
         if plural is not None:
-            # This should be a compiled function of a typical plural-form:
-            # Plural-Forms: nplurals=3; plural=n%10==1 && n%100!=11 ? 0 :
-            #               n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2;
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