# Bug Report: pandas.core.computation.eval Inconsistent Empty Expression Handling

**Target**: `pandas.core.computation.eval.eval`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `eval` function inconsistently handles empty vs whitespace-only expressions. Empty strings raise `ValueError`, but whitespace-only strings silently return `None`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.computation.eval import eval
import pytest

@given(st.text())
def test_empty_expressions_should_raise(s):
    if not s.strip():
        with pytest.raises(ValueError, match="expr cannot be an empty string"):
            eval(s)
```

**Failing input**: `' '` (single space)

## Reproducing the Bug

```python
from pandas.core.computation.eval import eval

print(eval(""))
print(eval("   "))
print(eval("\n"))
```

Output:
```
ValueError: expr cannot be an empty string
None
None
```

## Why This Is A Bug

The function `_check_expression` is designed to reject empty expressions, as evidenced by its error message "expr cannot be an empty string". However, it only checks `not expr`, which is False for whitespace-only strings. This creates an inconsistency where `eval("")` raises an error but `eval("   ")` silently returns `None`. A whitespace-only string contains no meaningful expression and should be treated the same as an empty string.

## Fix

```diff
--- a/pandas/core/computation/eval.py
+++ b/pandas/core/computation/eval.py
@@ -119,7 +119,8 @@ def _check_expression(expr):
     ValueError
       * If expr is an empty string
     """
-    if not expr:
+    s = expr if isinstance(expr, str) else str(expr)
+    if not s or not s.strip():
         raise ValueError("expr cannot be an empty string")
```