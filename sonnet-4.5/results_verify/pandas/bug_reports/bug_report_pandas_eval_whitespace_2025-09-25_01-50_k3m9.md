# Bug Report: pandas.core.computation.eval Whitespace-Only Expression Validation

**Target**: `pandas.core.computation.eval.eval`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`pd.eval()` accepts whitespace-only expressions (e.g., `"   \n\t  "`) and returns `None` instead of raising a `ValueError`, violating the documented contract that empty expressions should raise an error.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import pytest

@given(st.text(min_size=1).filter(lambda s: s.strip() == ""))
def test_eval_whitespace_only_expression_should_raise(whitespace_expr):
    with pytest.raises(ValueError, match="expr cannot be an empty string"):
        pd.eval(whitespace_expr)
```

**Failing input**: `"   \n\t  "` (or any whitespace-only string)

## Reproducing the Bug

```python
import pandas as pd

result = pd.eval("   \n\t  ")
print(f"Result: {result}")

try:
    pd.eval("")
except ValueError as e:
    print(f"Empty string correctly raises: {e}")
```

Output:
```
Result: None
Empty string correctly raises: expr cannot be an empty string
```

## Why This Is A Bug

The docstring for `_check_expression` states:
```python
def _check_expression(expr):
    """
    Make sure an expression is not an empty string

    Raises
    ------
    ValueError
      * If expr is an empty string
    """
```

While `pd.eval("")` correctly raises `ValueError`, `pd.eval("   \n\t  ")` silently returns `None`. This is inconsistent behavior because whitespace-only expressions are semantically empty after processing.

The bug violates the API contract: users expect that invalid/empty expressions raise an error, not return `None`.

## Fix

The issue is in `eval.py` line 307:

```python
if isinstance(expr, str):
    _check_expression(expr)
    exprs = [e.strip() for e in expr.splitlines() if e.strip() != ""]
```

The check happens before stripping, so `"   \n\t  "` passes. After splitting and stripping, `exprs` becomes `[]`, which causes the function to return `None` without executing any loop iterations.

**Fix**: Add a check after processing the expression list:

```diff
--- a/pandas/core/computation/eval.py
+++ b/pandas/core/computation/eval.py
@@ -305,6 +305,8 @@ def eval(
     if isinstance(expr, str):
         _check_expression(expr)
         exprs = [e.strip() for e in expr.splitlines() if e.strip() != ""]
+        if not exprs:
+            raise ValueError("expr cannot be an empty string")
     else:
         # ops.BinOp; for internal compat, not intended to be passed by users
         exprs = [expr]
```