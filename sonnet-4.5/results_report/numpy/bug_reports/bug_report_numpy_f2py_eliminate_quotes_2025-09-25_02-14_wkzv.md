# Bug Report: numpy.f2py.symbolic.eliminate_quotes AssertionError on Unpaired Quotes

**Target**: `numpy.f2py.symbolic.eliminate_quotes`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `eliminate_quotes` function crashes with an AssertionError when given input strings containing unpaired quote characters, which can legitimately occur in Fortran source code (e.g., in comments).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from numpy.f2py import symbolic


@given(st.text(min_size=1, max_size=200))
@settings(max_examples=500)
def test_quote_elimination_round_trip(s):
    new_s, mapping = symbolic.eliminate_quotes(s)
    reconstructed = symbolic.insert_quotes(new_s, mapping)
    assert s == reconstructed
```

**Failing input**: `s='"'` (single double-quote character)

## Reproducing the Bug

```python
from numpy.f2py import symbolic

s = '"'
new_s, mapping = symbolic.eliminate_quotes(s)
```

Output:
```
AssertionError
```

The same error occurs with a single single-quote: `s = "'"`.

## Why This Is A Bug

The function's docstring states it "Replace[s] quoted substrings of input string" without documenting any preconditions about balanced quotes. The function uses a regex that only matches properly paired quotes (`"..."` or `'...'`), leaving unpaired quotes in the result. It then asserts that no quotes remain in the output, which fails.

This is problematic because:
1. Fortran source code can legitimately contain unpaired quotes (e.g., in comments: `! This is a comment with an unmatched "`)
2. The function should either handle unpaired quotes gracefully or raise a proper ValueError with a clear error message
3. AssertionError is an internal implementation detail that shouldn't be exposed to users

## Fix

The function should check for unpaired quotes after the regex substitution and either:
1. Leave them as-is (if they're not part of a string literal), or
2. Raise a descriptive ValueError

```diff
--- a/symbolic.py
+++ b/symbolic.py
@@ -1191,8 +1191,8 @@ def eliminate_quotes(s):
         double_quoted=r'("([^"\\]|(\\.))*")'),
         repl, s)

-    assert '"' not in new_s
-    assert "'" not in new_s
+    if '"' in new_s or "'" in new_s:
+        raise ValueError(f"Unpaired quote character found in input: {s!r}")

     return new_s, d
```