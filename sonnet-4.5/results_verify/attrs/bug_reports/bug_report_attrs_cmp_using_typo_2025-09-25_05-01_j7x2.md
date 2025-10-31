# Bug Report: attrs cmp_using Error Message Typo

**Target**: `attrs.cmp_using`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `cmp_using()` function has a grammatical error in its error message when `eq` is not provided but ordering functions are.

## Property-Based Test

```python
import attrs
from attrs import cmp_using
import pytest

def test_cmp_using_error_message():
    with pytest.raises(ValueError, match="eq must be define"):
        cmp_using(lt=lambda a, b: a < b)
```

**Failing input**: Any call to `cmp_using` with ordering functions but no `eq`.

## Reproducing the Bug

```python
from attrs import cmp_using

try:
    cmp_using(lt=lambda a, b: a < b)
except ValueError as e:
    print(e)
```

Output:
```
eq must be define is order to complete ordering from lt, le, gt, ge.
```

Expected:
```
eq must be defined in order to complete ordering from lt, le, gt, ge.
```

## Why This Is A Bug

The error message in `attr/_cmp.py` line 105 contains two grammatical errors:

1. "define" should be "defined"
2. "is order" should be "in order"

This violates professional quality standards for error messages.

## Fix

```diff
--- a/attr/_cmp.py
+++ b/attr/_cmp.py
@@ -102,7 +102,7 @@ def cmp_using(
         if not has_eq_function:
             # functools.total_ordering requires __eq__ to be defined,
             # so raise early error here to keep a nice stack.
-            msg = "eq must be define is order to complete ordering from lt, le, gt, ge."
+            msg = "eq must be defined in order to complete ordering from lt, le, gt, ge."
             raise ValueError(msg)
         type_ = functools.total_ordering(type_)
```