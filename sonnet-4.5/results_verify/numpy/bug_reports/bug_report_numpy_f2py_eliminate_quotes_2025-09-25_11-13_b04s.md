# Bug Report: numpy.f2py.symbolic.eliminate_quotes Assertion Failure on Unterminated Quotes

**Target**: `numpy.f2py.symbolic.eliminate_quotes`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`eliminate_quotes()` crashes with AssertionError when given a string containing unterminated quote characters, violating its claimed round-trip property with `insert_quotes()`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.f2py.symbolic as symbolic

@given(st.text())
def test_quotes_round_trip(s):
    new_s, d = symbolic.eliminate_quotes(s)
    reconstructed = symbolic.insert_quotes(new_s, d)
    assert reconstructed == s
```

**Failing input**: `s='"'` and `s="'"`

## Reproducing the Bug

```python
import numpy.f2py.symbolic as symbolic

s = '"'
new_s, d = symbolic.eliminate_quotes(s)
```

## Why This Is A Bug

The function's regex only matches properly closed quoted strings. Unterminated quotes like `"` or `'` are left in the output string `new_s`, which then fails the post-condition assertion `assert '"' not in new_s`. The function should either:
1. Handle unterminated quotes gracefully (skip them or replace them)
2. Raise a proper exception with an informative error message
3. Document that it requires well-formed quoted strings

## Fix

The assertions should be replaced with better error handling:

```diff
--- a/numpy/f2py/symbolic.py
+++ b/numpy/f2py/symbolic.py
@@ -1191,8 +1191,11 @@ def eliminate_quotes(s):
         double_quoted=r'("([^"\\]|(\\.))*")'),
         repl, s)

-    assert '"' not in new_s
-    assert "'" not in new_s
+    if '"' in new_s or "'" in new_s:
+        raise ValueError(
+            f"String contains unterminated quotes: {s!r}"
+        )

     return new_s, d
```