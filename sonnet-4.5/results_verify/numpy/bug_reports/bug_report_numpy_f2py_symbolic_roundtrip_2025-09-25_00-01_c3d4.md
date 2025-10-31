# Bug Report: numpy.f2py.symbolic Round-trip Equality Failure

**Target**: `numpy.f2py.symbolic.Expr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse/tostring` round-trip violates equality for certain expressions involving division with negative denominators, causing `parse(expr.tostring()) != expr` even though the expressions are semantically equivalent.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy.f2py.symbolic as symbolic

simple_expr_st = st.one_of(
    st.from_regex(r'[a-zA-Z_][a-zA-Z0-9_]*', fullmatch=True),
    st.integers(min_value=-1000000, max_value=1000000).map(str),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6).map(lambda f: f"{f:.6g}"),
)

@st.composite
def expr_strings(draw):
    left = draw(simple_expr_st)
    right = draw(simple_expr_st)
    op = draw(st.sampled_from(['+', '-', '*', '/']))
    return f"{left} {op} {right}"

@given(expr_strings())
@settings(max_examples=500)
def test_parse_tostring_roundtrip(expr_str):
    expr = symbolic.Expr.parse(expr_str)
    s = expr.tostring()
    expr2 = symbolic.Expr.parse(s)
    assert expr == expr2
```

**Failing input**: `'0 / -1'`

## Reproducing the Bug

```python
import numpy.f2py.symbolic as symbolic

expr1 = symbolic.Expr.parse('0 / -1')
s = expr1.tostring()
expr2 = symbolic.Expr.parse(s)

assert expr1 == expr2
```

Output:
```
AssertionError: Round-trip failed: parse(expr.tostring()) != expr
```

Investigation shows:
```python
print(repr(expr1))
# Expr(Op.TERMS, {Expr(Op.APPLY, (<ArithOp.DIV: 6>, (Expr(Op.INTEGER, (0, 4)), Expr(Op.SYMBOL, '')), {})): 1, Expr(Op.INTEGER, (1, 4)): -1})

print(repr(expr2))
# Expr(Op.TERMS, {Expr(Op.INTEGER, (1, 4)): -1, Expr(Op.APPLY, (Expr(Op.SYMBOL, 'ArithOp.DIV'), (Expr(Op.INTEGER, (0, 4)), Expr(Op.SYMBOL, '')), {})): 1})
```

The key difference: `expr1` has `<ArithOp.DIV: 6>` (the actual enum value), while `expr2` has `Expr(Op.SYMBOL, 'ArithOp.DIV')` (a symbol with the name).

## Why This Is A Bug

1. The fundamental contract of `parse/tostring` is that they should be inverses: `parse(expr.tostring()) == expr`
2. The `tostring()` method produces output like `'-1 + ArithOp.DIV(0, )'` which includes the internal representation `ArithOp.DIV`
3. When re-parsed, this gets interpreted as a symbol rather than the operator, breaking equality
4. While the expressions may be semantically equivalent, violating equality breaks hashability and usage in sets/dicts

## Fix

The issue is in how `tostring()` represents `ArithOp.DIV` applications. The method should either:

1. Output valid parseable syntax (e.g., `'0 / -1'` instead of `'-1 + ArithOp.DIV(0, )'`), or
2. Parse `ArithOp.DIV(...)` syntax correctly to reconstruct the proper operator

The cleaner fix is option 1 - ensure `tostring()` outputs valid expression syntax that can be parsed back:

```diff
--- a/numpy/f2py/symbolic.py
+++ b/numpy/f2py/symbolic.py
@@ -xxx,x +xxx,x @@
 def tostring(self, parent_precedence=Precedence.NONE, language=Language.Fortran):
     # For ArithOp.DIV applications, output standard division syntax
+    if self.op == Op.APPLY and self.data[0] == ArithOp.DIV:
+        args = self.data[1]
+        if len(args) == 2:
+            return f"({args[0].tostring()}) / ({args[1].tostring()})"
     # ... existing tostring logic
```

Note: The exact line numbers would need to be determined from the source, but the fix involves ensuring `tostring()` outputs parseable syntax for division operations.