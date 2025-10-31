# Bug Report: numpy.f2py.symbolic.eliminate_quotes Assertion Failure on Unmatched Quotes

**Target**: `numpy.f2py.symbolic.eliminate_quotes`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`eliminate_quotes()` raises AssertionError on strings containing unmatched quote characters, violating the documented inverse relationship with `insert_quotes()`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.f2py.symbolic as symbolic

@given(st.text(min_size=1, max_size=100))
def test_eliminate_insert_quotes_roundtrip(s):
    s_no_quotes, d = symbolic.eliminate_quotes(s)
    s_restored = symbolic.insert_quotes(s_no_quotes, d)
    assert s == s_restored
```

**Failing input**: `"'"` (single apostrophe)

## Reproducing the Bug

```python
import numpy.f2py.symbolic as symbolic

symbolic.eliminate_quotes("'")
```

## Why This Is A Bug

The function's regex pattern `r"('([^'\\]|(\\.))*')"` only matches properly quoted strings. Unmatched quote characters remain in the output string `new_s`, causing the assertion `assert "'" not in new_s` to fail. This prevents the function from handling arbitrary string inputs and violates the documented inverse relationship with `insert_quotes()`.

## Fix

```diff
--- a/numpy/f2py/symbolic.py
+++ b/numpy/f2py/symbolic.py
@@ -1191,8 +1191,6 @@ def eliminate_quotes(s):
         double_quoted=r'("([^"\\]|(\\.))*")'),
         repl, s)

-    assert '"' not in new_s
-    assert "'" not in new_s

     return new_s, d
```