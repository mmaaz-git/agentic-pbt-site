# Bug Report: pd.eval Silently Accepts Whitespace-Only Expressions

**Target**: `pandas.core.computation.eval._check_expression` and `pandas.core.computation.api.eval`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `pd.eval` function silently returns `None` when given whitespace-only expressions (e.g., `" "`, `"\t"`, `"\n"`), instead of raising a `ValueError` like it does for empty strings.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.computation.eval import _check_expression
import pytest

@given(st.text().filter(lambda x: x.strip() == "" and x != ""))
def test_check_expression_raises_on_whitespace(expr):
    with pytest.raises(ValueError, match="expr cannot be an empty string"):
        _check_expression(expr)
```

**Failing input**: `" "` (single space)

## Reproducing the Bug

```python
import pandas as pd

result = pd.eval(" ")
print(f"pd.eval(' ') returned: {repr(result)}")
```

Output:
```
pd.eval(' ') returned: None
```

Compare with:
```python
import pandas as pd

result = pd.eval("")
```

Output:
```
ValueError: expr cannot be an empty string
```

## Why This Is A Bug

The `_check_expression` function's docstring states: "Make sure an expression is not an empty string" and raises `ValueError("expr cannot be an empty string")`. However, it only checks `if not expr`, which evaluates to `False` for whitespace strings like `" "` (since `bool(" ") == True`).

Whitespace-only strings are semantically empty expressions and should be rejected just like truly empty strings. The current behavior silently returns `None`, which is inconsistent and potentially confusing for users who might expect an error.

## Fix

```diff
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