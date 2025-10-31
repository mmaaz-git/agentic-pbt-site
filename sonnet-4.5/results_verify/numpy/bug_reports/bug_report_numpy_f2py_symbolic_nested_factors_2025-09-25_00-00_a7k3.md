# Bug Report: numpy.f2py.symbolic Nested Factors Not Normalized

**Target**: `numpy.f2py.symbolic.normalize`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalize` function fails to simplify nested `FACTORS` expressions with negative exponents, violating the mathematical property that `(x ** a) ** b = x ** (a*b)`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from numpy.f2py.symbolic import as_number, normalize


@st.composite
def expr_integers(draw):
    value = draw(st.integers(min_value=-100, max_value=100))
    return as_number(value, kind=4)


@given(expr_integers(), st.integers(min_value=-5, max_value=-1), st.integers(min_value=-5, max_value=-1))
@settings(max_examples=500)
def test_power_associativity_negative_exponents(expr, m, n):
    assume(expr.data[0] != 0)
    result = (expr ** m) ** n
    expected = expr ** (m * n)
    assert normalize(result) == normalize(expected)
```

**Failing input**: `expr=Expr(Op.INTEGER, (5, 4)), m=-1, n=-1`

## Reproducing the Bug

```python
from numpy.f2py.symbolic import as_number, normalize

expr = as_number(5, 4)

result = (expr ** -1) ** -1
expected = expr

print(f"Result: {repr(normalize(result))}")
print(f"Expected: {repr(normalize(expected))}")
```

**Output:**
```
Result: Expr(Op.FACTORS, {Expr(Op.FACTORS, {Expr(Op.INTEGER, (5, 4)): -1}): -1})
Expected: Expr(Op.INTEGER, (5, 4))
```

## Why This Is A Bug

The mathematical property `(x ** a) ** b = x ** (a*b)` should always hold. When `a=-1` and `b=-1`, we should get `x ** 1 = x`. However, the `normalize` function creates a nested `FACTORS` structure instead of simplifying to the base expression.

This happens because the normalization code in `symbolic.py:836-841` only flattens nested factors when the exponent is positive (`e > 0`), but fails to flatten when the exponent is negative or zero.

## Fix

```diff
--- a/numpy/f2py/symbolic.py
+++ b/numpy/f2py/symbolic.py
@@ -834,7 +834,7 @@ def normalize(obj):
                 else:
                     _pairs_add(d, b, e)
             elif b.op is Op.FACTORS:
-                if e > 0 and isinstance(e, integer_types):
+                if isinstance(e, integer_types):
                     for b1, e1 in b.data.items():
                         _pairs_add(d, b1, e1 * e)
                 else:
```