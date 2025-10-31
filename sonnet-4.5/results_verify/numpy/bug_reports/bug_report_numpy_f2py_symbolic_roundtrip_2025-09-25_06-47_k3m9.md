# Bug Report: numpy.f2py.symbolic Round-trip Failure for Division and Relational Operators

**Target**: `numpy.f2py.symbolic.Expr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Expr.parse()` and `__str__()` methods violate the round-trip property for division and relational operators due to mismatched default languages between parsing (C) and serialization (Fortran).

## Property-Based Test

```python
import numpy.f2py.symbolic as symbolic
from hypothesis import given, settings, strategies as st

expr_parse = symbolic.Expr.parse

simple_expr_strategy = st.sampled_from([
    'x', 'y', 'z', 'a', 'b', 'c',
    '1', '2', '3', '0',
    'x + y', 'x - y', 'x * y', 'x / y',
    'a + b', 'a * b', 'a - b', 'a / b',
    'x + 1', 'x * 2', 'x - 3',
    '(x + y) * z', 'x * (y + z)',
    'a + b + c', 'a * b * c',
    '2 * x + 3', '(a + b) * (c + d)',
    '-x', '-a',
    'x == y', 'x < y', 'x > y',
])

@given(simple_expr_strategy)
@settings(max_examples=500)
def test_parse_str_roundtrip(expr_str):
    e1 = expr_parse(expr_str)
    s = str(e1)
    e2 = expr_parse(s)
    assert e1 == e2, f"Round-trip failed: parse('{expr_str}') -> str -> parse != original"
```

**Failing input**: `'x / y'`

## Reproducing the Bug

```python
import numpy.f2py.symbolic as symbolic

expr_parse = symbolic.Expr.parse

e1 = expr_parse('x / y')
s = str(e1)
e2 = expr_parse(s)

print(f'Original: {repr(e1)}')
print(f'Stringified: "{s}"')
print(f'Reparsed: {repr(e2)}')
print(f'Equal: {e1 == e2}')

assert e1 == e2
```

Output:
```
Original: Expr(Op.APPLY, (<ArithOp.DIV: 6>, (Expr(Op.SYMBOL, 'x'), Expr(Op.SYMBOL, 'y')), {}))
Stringified: "ArithOp.DIV(x, y)"
Reparsed: Expr(Op.APPLY, (Expr(Op.SYMBOL, 'ArithOp.DIV'), (Expr(Op.SYMBOL, 'x'), Expr(Op.SYMBOL, 'y')), {}))
Equal: False
AssertionError
```

## Why This Is A Bug

The root cause is a language mismatch between `parse()` and `tostring()`:
- `Expr.parse()` defaults to `language=Language.C`
- `Expr.tostring()` (used by `__str__()`) defaults to `language=Language.Fortran`

When parsing with C syntax and serializing with Fortran syntax, the round-trip fails:

1. `parse('x / y', language=C)` creates a division expression with `ArithOp.DIV` enum as the operator
2. `str(expr)` calls `tostring(language=Fortran)` → outputs `"ArithOp.DIV(x, y)"`
3. `parse("ArithOp.DIV(x, y)", language=C)` interprets `'ArithOp.DIV'` as a symbol string, not the enum

The same bug affects all relational operators:
- `x == y` → `"x .eq. y"` (C parser doesn't recognize Fortran `.eq.` syntax)
- `x < y` → `"x .lt. y"`
- `x > y` → `"x .gt. y"`
- `x <= y` → `"x .le. y"`
- `x >= y` → `"x .ge. y"`
- `x != y` → `"x .ne. y"`

This violates the fundamental round-trip property `parse(str(expr)) == expr` and breaks serialization/deserialization workflows. The parser even emits warnings: `ExprWarning: fromstring: treating 'ArithOp.DIV' as symbol`.

## Fix

The simplest fix is to make the default languages consistent:

```diff
--- a/numpy/f2py/symbolic.py
+++ b/numpy/f2py/symbolic.py
@@ -tostring method signature
-    def tostring(self, parent_precedence=Precedence.NONE, language=Language.Fortran):
+    def tostring(self, parent_precedence=Precedence.NONE, language=Language.C):
```

This ensures `parse(str(expr))` uses C syntax for both operations, fixing the round-trip property. With this change, division would serialize as `"x / y"` instead of `"ArithOp.DIV(x, y)"`, and relational operators would use C syntax (`==`, `<`, etc.) instead of Fortran syntax (`.eq.`, `.lt.`, etc.).