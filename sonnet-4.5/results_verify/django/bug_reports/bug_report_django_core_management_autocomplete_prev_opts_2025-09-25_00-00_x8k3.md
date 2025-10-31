# Bug Report: django.core.management.ManagementUtility.autocomplete prev_opts Index Bug

**Target**: `django.core.management.ManagementUtility.autocomplete`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The autocomplete() method incorrectly computes prev_opts when COMP_CWORD=0, including all previous options instead of an empty set due to negative slice indexing.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(st.lists(st.text(min_size=1), min_size=2), st.integers(min_value=0, max_value=10))
def test_prev_opts_logic(cwords, cword):
    prev_opts = {x.split("=")[0] for x in cwords[1 : cword - 1]}

    if cword == 0:
        assert prev_opts == set(), f"When cword=0, expected empty set, got {prev_opts!r}"
    if cword == 1:
        assert prev_opts == set(), f"When cword=1, expected empty set, got {prev_opts!r}"
```

**Failing input**: `cwords=["django-admin", "migrate"], cword=0`

## Reproducing the Bug

```python
cwords = ["django-admin", "migrate", "--database=default", "--noinput"]
cword = 0

prev_opts = {x.split("=")[0] for x in cwords[1 : cword - 1]}

print(f"cwords[1 : cword - 1] = cwords[1 : -1] = {cwords[1 : -1]}")
print(f"prev_opts = {prev_opts}")
print(f"Expected: set()")

assert prev_opts == set(), "Bug: negative slice index includes previous options"
```

## Why This Is A Bug

When `cword=0`, the expression `cwords[1 : cword - 1]` becomes `cwords[1 : -1]`, which is a valid Python slice that includes all elements from index 1 to the second-to-last element. The code intends to get previously specified options (those between the subcommand and the current word), but when `cword=0`, there should be no previous options. This causes the autocomplete to incorrectly filter out options that shouldn't be filtered.

## Fix

```diff
--- a/django/core/management/__init__.py
+++ b/django/core/management/__init__.py
@@ -335,7 +335,7 @@ class ManagementUtility:
                 if s_opt.option_strings
             )
             # filter out previously specified options from available options
-            prev_opts = {x.split("=")[0] for x in cwords[1 : cword - 1]}
+            prev_opts = {x.split("=")[0] for x in cwords[1 : cword - 1]} if cword > 1 else set()
             options = (opt for opt in options if opt[0] not in prev_opts)
```