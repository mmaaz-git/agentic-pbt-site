# Bug Report: numpy.f2py.symbolic.eliminate_quotes AssertionError on Unmatched Quotes

**Target**: `numpy.f2py.symbolic.eliminate_quotes`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `eliminate_quotes` function crashes with an AssertionError when given a string containing a single unmatched quote character (`"` or `'`), rather than raising a proper exception or handling the input gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.f2py.symbolic as symbolic

@given(st.text())
def test_eliminate_insert_quotes_roundtrip(s):
    new_s, mapping = symbolic.eliminate_quotes(s)
    restored = symbolic.insert_quotes(new_s, mapping)
    assert restored == s
```

**Failing inputs**:
- `s='"'`
- `s="'"`

## Reproducing the Bug

```python
import numpy.f2py.symbolic as symbolic

symbolic.eliminate_quotes('"')
```

This raises:
```
AssertionError
```

Similarly:
```python
symbolic.eliminate_quotes("'")
```

## Why This Is A Bug

The function uses assertions (`assert '"' not in new_s` and `assert "'" not in new_s`) to validate its output. When given an unmatched quote character, the regex doesn't match it (since it only matches properly quoted strings), so the quote remains in `new_s`, triggering the assertion.

Using assertions for input validation is problematic because:
1. Assertions can be disabled with Python's `-O` flag
2. AssertionError is not a semantic error type for invalid input
3. The function should either handle unmatched quotes gracefully or raise ValueError

## Fix

The function should raise a proper ValueError when it detects unmatched quotes in the input, rather than using assertions:

```diff
--- a/symbolic.py
+++ b/symbolic.py
@@ -1191,8 +1191,10 @@ def eliminate_quotes(s):
         double_quoted=r'("([^"\\]|(\\.))*")'),
         repl, s)

-    assert '"' not in new_s
-    assert "'" not in new_s
+    if '"' in new_s or "'" in new_s:
+        raise ValueError(f'Unmatched quote in input string: {s!r}')

     return new_s, d
```