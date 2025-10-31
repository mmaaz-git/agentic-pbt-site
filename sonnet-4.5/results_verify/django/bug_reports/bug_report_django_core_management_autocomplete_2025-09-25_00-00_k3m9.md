# Bug Report: django.core.management.ManagementUtility.autocomplete Index Bug

**Target**: `django.core.management.ManagementUtility.autocomplete`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The autocomplete() method incorrectly handles the case when COMP_CWORD=0 due to negative indexing behavior in Python, causing it to access the last element of the list instead of setting curr to an empty string.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=10), st.integers())
def test_autocomplete_index_logic(cwords, cword):
    try:
        curr = cwords[cword - 1]
    except IndexError:
        curr = ""

    if cword == 0 and len(cwords) > 0:
        assert curr == "", f"When cword=0, expected curr='', but got {curr!r}"
```

**Failing input**: `cwords=['0'], cword=0`

## Reproducing the Bug

```python
cwords = ["django-admin", "migrate"]
cword = 0

try:
    curr = cwords[cword - 1]
except IndexError:
    curr = ""

print(f"curr = {curr!r}")
print(f"Expected: ''")
print(f"Actual: 'migrate'")
assert curr == "", "Bug: negative index accesses last element"
```

## Why This Is A Bug

When `cword=0`, the expression `cword - 1 = -1`, which is a valid Python index that accesses the last element of the list. The code intends to catch IndexError and set `curr=""` for out-of-bounds access, but Python's negative indexing prevents the IndexError from being raised. This causes incorrect autocomplete behavior when the cursor is at the first position in bash completion.

## Fix

```diff
--- a/django/core/management/__init__.py
+++ b/django/core/management/__init__.py
@@ -304,7 +304,7 @@ class ManagementUtility:
         cword = int(os.environ["COMP_CWORD"])

         try:
-            curr = cwords[cword - 1]
+            curr = cwords[cword - 1] if cword > 0 else ""
         except IndexError:
             curr = ""
```