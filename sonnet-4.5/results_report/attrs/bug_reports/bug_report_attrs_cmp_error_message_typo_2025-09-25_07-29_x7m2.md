# Bug Report: attrs cmp_using Error Message Typo

**Target**: `attr._cmp.cmp_using`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `cmp_using()` function has a grammatical error in its error message when `eq` is not provided with ordering functions.

## Property-Based Test

This bug was discovered through code inspection while analyzing error messages.

```python
from hypothesis import given, strategies as st
from attr._cmp import cmp_using

@given(st.sampled_from([lambda a, b: a < b, lambda a, b: a > b]))
def test_cmp_using_error_message_grammar(lt_func):
    try:
        cmp_using(lt=lt_func)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        assert "must be defined in order" in error_msg, \
            f"Error message has typos: {error_msg}"
```

**Failing input**: Any ordering function without `eq`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

from attr._cmp import cmp_using

try:
    cmp_using(lt=lambda a, b: a < b)
except ValueError as e:
    print(f"Error message: {e}")
```

**Output**: `eq must be define is order to complete ordering from lt, le, gt, ge.`

## Why This Is A Bug

The error message at `attr/_cmp.py:105` contains two grammatical errors:

1. "define" should be "defined" (missing 'd')
2. "is order" should be "in order"

The message should read: "eq must be defined in order to complete ordering from lt, le, gt, ge."

This violates professional standards for error messages and could confuse users.

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