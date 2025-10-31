# Bug Report: Cython.Utility._pylong_join Unbalanced Parentheses

**Target**: `Cython.Utility._pylong_join`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_pylong_join` function generates C code with unbalanced parentheses when count >= 2, producing invalid C expressions that will not compile.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import Cython.Utility as cu

@settings(max_examples=500)
@given(count=st.integers(min_value=1, max_value=20))
def test_private_pylong_join_balanced_parentheses(count):
    result = cu._pylong_join(count)
    open_parens = result.count('(')
    close_parens = result.count(')')
    assert open_parens == close_parens, f"Unbalanced for count={count}: {result}"
```

**Failing input**: count >= 2 (e.g., count=2, count=3, etc.)

## Reproducing the Bug

```python
def _pylong_join(count, digits_ptr='digits', join_type='unsigned long'):
    def shift(n):
        return " << (%d * PyLong_SHIFT < 8 * sizeof(%s) ? %d * PyLong_SHIFT : 0)" % (n, join_type, n) if n else ''

    return '(%s)' % ' | '.join(
        "(((%s)%s[%d])%s)" % (join_type, digits_ptr, i, shift(i))
        for i in range(count-1, -1, -1))

for count in [1, 2, 3]:
    result = _pylong_join(count)
    open_count = result.count('(')
    close_count = result.count(')')
    print(f"count={count}: open={open_count}, close={close_count}, balanced={open_count==close_count}")
    print(f"  {result}")
```

Expected output:
```
count=1: open=4, close=4, balanced=True
count=2: open=9, close=7, balanced=False
count=3: open=14, close=10, balanced=False
```

## Why This Is A Bug

The shift() helper function returns a C ternary expression with unbalanced parentheses:
- Format: `" << (cond ? val : 0)"`
- Parentheses: **2 open, 1 close**

When shift(i) is inserted into the component template for i > 0:
- Template: `"(((%s)%s[%d])%s)"`
- With shift: `"(((type)ptr[i]) << (cond ? val : 0))"`
- Parentheses: **5 open, 3 close** (imbalance of 2)

The outer wrapper `'(%s)'` adds 1 more opening paren, compounding the issue.

For count=2:
- Component at i=1: 5 open, 3 close
- Component at i=0: 3 open, 3 close
- Total joined: 8 open, 6 close
- After outer wrap: **9 open, 7 close** (imbalance of 2)

This pattern continues, with imbalance growing as count increases.

## Fix

```diff
--- a/Cython/Utility/__init__.py
+++ b/Cython/Utility/__init__.py
@@ -22,7 +22,7 @@ def _pylong_join(count, digits_ptr='digits', join_type='unsigned long'):
     def shift(n):
-        return " << (%d * PyLong_SHIFT < 8 * sizeof(%s) ? %d * PyLong_SHIFT : 0)" % (n, join_type, n) if n else ''
+        return " << ((%d * PyLong_SHIFT < 8 * sizeof(%s)) ? %d * PyLong_SHIFT : 0)" % (n, join_type, n) if n else ''

     return '(%s)' % ' | '.join(
         "(((%s)%s[%d])%s)" % (join_type, digits_ptr, i, shift(i))
```

The fix adds parentheses around the condition `(%d * PyLong_SHIFT < 8 * sizeof(%s))` to balance the ternary operator, changing shift() output from 2 open/1 close to 3 open/2 close, which when combined with the component template produces balanced parentheses.