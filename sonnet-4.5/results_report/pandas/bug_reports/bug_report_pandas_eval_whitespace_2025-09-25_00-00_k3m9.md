# Bug Report: pandas.core.computation.eval Whitespace-Only Expression

**Target**: `pandas.core.computation.eval._check_expression`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `pandas.eval()` function rejects empty strings with a ValueError but silently accepts whitespace-only strings (e.g., `" "`, `"\t"`, `"\n"`, `"\r"`), returning `None` instead. This is inconsistent behavior that violates the function's documented contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.computation.eval import _check_expression

@given(st.just("") | st.text().filter(lambda x: x.isspace()))
def test_check_expression_rejects_empty_string(expr):
    with pytest.raises(ValueError, match="expr cannot be an empty string"):
        _check_expression(expr)
```

**Failing input**: `'\r'` (and any whitespace-only string)

## Reproducing the Bug

```python
import pandas as pd

pd.eval("")
pd.eval(" ")
pd.eval("\t")
pd.eval("\n")
pd.eval("\r")
```

Output:
```
ValueError: expr cannot be an empty string
None
None
None
None
```

## Why This Is A Bug

1. **Inconsistent behavior**: Empty string `""` raises `ValueError`, but whitespace-only strings like `" "` return `None`
2. **Violates documented contract**: The docstring for `_check_expression` states it should "Make sure an expression is not an empty string", but whitespace-only strings are semantically empty
3. **Unexpected None return**: Users expect either a valid result or an exception, not a silent `None` return
4. **Code analysis**: After `_check_expression`, the code does:
   ```python
   exprs = [e.strip() for e in expr.splitlines() if e.strip() != ""]
   ```
   For whitespace-only inputs, this produces an empty list, causing the function to return `None` without any computation or error

## Fix

```diff
diff --git a/pandas/core/computation/eval.py b/pandas/core/computation/eval.py
index 1234567..abcdefg 100644
--- a/pandas/core/computation/eval.py
+++ b/pandas/core/computation/eval.py
@@ -119,7 +119,7 @@ def _check_expression(expr):
     ValueError
       * If expr is an empty string
     """
-    if not expr:
+    if not expr or not expr.strip():
         raise ValueError("expr cannot be an empty string")
```