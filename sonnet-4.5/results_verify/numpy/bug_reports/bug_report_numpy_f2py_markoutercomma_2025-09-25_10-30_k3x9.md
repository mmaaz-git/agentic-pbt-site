# Bug Report: numpy.f2py.crackfortran.markoutercomma Assertion Failure

**Target**: `numpy.f2py.crackfortran.markoutercomma`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `markoutercomma` function crashes with an AssertionError when given strings with unbalanced parentheses, providing an unclear error message for a public API function.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy.f2py.crackfortran as crackfortran


@given(st.text())
@settings(max_examples=1000)
def test_markoutercomma_should_not_crash(line):
    try:
        crackfortran.markoutercomma(line)
    except AssertionError as e:
        assert False, f"Crashed with AssertionError: {e}"
```

**Failing input**: `')'`

## Reproducing the Bug

```python
import numpy.f2py.crackfortran as crackfortran

result = crackfortran.markoutercomma(')')
```

Output:
```
AssertionError: (-1, ')', ')')
```

## Why This Is A Bug

The `markoutercomma` function is a public function in the `numpy.f2py.crackfortran` module (no leading underscore). When called with a string containing unbalanced parentheses, it crashes with a cryptic assertion error `(-1, ')', ')')` instead of either:

1. Handling the input gracefully by raising a proper ValueError with a descriptive message, or
2. Being marked as a private function (with a leading underscore) to indicate it's not meant for external use

The assertion `assert not f` at line 889 is meant to catch programming errors during development, but for a public API function, this should be a proper input validation error.

## Fix

```diff
--- a/crackfortran.py
+++ b/crackfortran.py
@@ -886,7 +886,8 @@ def markoutercomma(line, comma=','):
                 f -= 1
         before, after = split_by_unquoted(after[1:], comma + '()')
         l += before
-    assert not f, repr((f, line, l))
+    if f != 0:
+        raise ValueError(f"Unbalanced parentheses in input: {line!r}")
     return l
```