# Bug Report: Cython.Utility pylong_join Inconsistency

**Target**: `Cython.Utility.pylong_join` and `Cython.Utility._pylong_join`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The two implementations of pylong_join in Cython.Utility produce inconsistent output for count=0 and negative values. `pylong_join` returns an empty string while `_pylong_join` returns `'()'`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Utility import pylong_join, _pylong_join


@given(st.integers(min_value=-10, max_value=20))
@settings(max_examples=1000)
def test_pylong_join_consistency(count):
    result1 = pylong_join(count)
    result2 = _pylong_join(count)

    assert result1 == result2, (
        f"Inconsistency between pylong_join and _pylong_join for count={count}:\n"
        f"  pylong_join:  {result1!r}\n"
        f"  _pylong_join: {result2!r}"
    )
```

**Failing input**: `count=0`

## Reproducing the Bug

```python
from Cython.Utility import pylong_join, _pylong_join

result1 = pylong_join(0)
result2 = _pylong_join(0)

print(f"pylong_join(0):  {result1!r}")
print(f"_pylong_join(0): {result2!r}")

assert result1 == result2
```

Output:
```
pylong_join(0):  ''
_pylong_join(0): '()'
AssertionError
```

## Why This Is A Bug

The comment on line 13-14 of `Cython/Utility/__init__.py` states that `_pylong_join` is "a bit slower than the simpler one above", implying it's an alternative implementation that should produce equivalent output. However, the two functions produce different results for count=0:

- `pylong_join(0)` returns `''` (empty string)
- `_pylong_join(0)` returns `'()'` (empty parentheses)

This inconsistency could cause bugs if code switches between these implementations or if one is used as a reference for the other.

## Fix

The issue is in `_pylong_join` on line 26. When the join produces an empty string, it still wraps it in `'(%s)'` format, resulting in `'()'`. The fix is to handle the empty case consistently with `pylong_join`:

```diff
--- a/Cython/Utility/__init__.py
+++ b/Cython/Utility/__init__.py
@@ -23,6 +23,8 @@ def _pylong_join(count, digits_ptr='digits', join_type='unsigned long'):
         # avoid compiler warnings for overly large shifts that will be discarded anyway
         return " << (%d * PyLong_SHIFT < 8 * sizeof(%s) ? %d * PyLong_SHIFT : 0)" % (n, join_type, n) if n else ''

-    return '(%s)' % ' | '.join(
-        "(((%s)%s[%d])%s)" % (join_type, digits_ptr, i, shift(i))
-        for i in range(count-1, -1, -1))
+    if count <= 0:
+        return ''
+    else:
+        return '(%s)' % ' | '.join(
+            "(((%s)%s[%d])%s)" % (join_type, digits_ptr, i, shift(i))
+            for i in range(count-1, -1, -1))
```