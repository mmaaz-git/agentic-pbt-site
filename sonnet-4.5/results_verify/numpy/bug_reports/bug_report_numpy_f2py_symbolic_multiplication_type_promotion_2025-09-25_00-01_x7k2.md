# Bug Report: numpy.f2py.symbolic Multiplication Type Promotion Inconsistency

**Target**: `numpy.f2py.symbolic.Expr.__mul__`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Multiplication operation produces different types (INTEGER vs REAL) depending on evaluation order when operands include zeros, violating associativity and type consistency.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from numpy.f2py.symbolic import as_symbol, as_integer, as_real, Expr, Op

@st.composite
def expr_integers(draw):
    value = draw(st.integers(min_value=-1000, max_value=1000))
    kind = draw(st.sampled_from([4, 8]))
    return as_integer(value, kind)

@st.composite
def expr_reals(draw):
    value = draw(st.floats(allow_nan=False, allow_infinity=False,
                           min_value=-1e6, max_value=1e6))
    kind = draw(st.sampled_from([4, 8]))
    return as_real(value, kind)

@st.composite
def expr_symbols(draw):
    name = draw(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10))
    return as_symbol(name)

@st.composite
def expr_trees(draw, max_depth=3):
    if max_depth == 0:
        choice = draw(st.sampled_from(['int', 'real', 'symbol']))
        if choice == 'int':
            return draw(expr_integers())
        elif choice == 'real':
            return draw(expr_reals())
        else:
            return draw(expr_symbols())

    choice = draw(st.sampled_from(['simple', 'mul']))
    if choice == 'simple':
        return draw(expr_trees(max_depth=0))
    else:
        left = draw(expr_trees(max_depth=max_depth-1))
        right = draw(expr_trees(max_depth=max_depth-1))
        return left * right

@given(expr_trees(), expr_trees(), expr_trees())
def test_multiplication_associative(a, b, c):
    assert (a * b) * c == a * (b * c)
```

**Failing input**: `a=Expr(Op.INTEGER, (0, 4))`, `b=Expr(Op.REAL, (1.0, 4))`, `c=Expr(Op.SYMBOL, 'a')`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages')

from numpy.f2py.symbolic import as_symbol, as_integer, as_real

x = as_integer(0, 4)
y = as_real(1.0, 4)
z = as_symbol('a')

left = (x * y) * z
right = x * (y * z)

print(f"(0 * 1.0) * a = {left} (op: {left.op})")
print(f"0 * (1.0 * a) = {right} (op: {right.op})")
print(f"Equal? {left == right}")
```

Output:
```
(0 * 1.0) * a = 0.0 (op: Op.REAL)
0 * (1.0 * a) = 0 (op: Op.INTEGER)
Equal? False
```

## Why This Is A Bug

Multiplication must be associative: `(a * b) * c` should equal `a * (b * c)` for all values. The bug occurs because:

1. When computing `(0 * 1.0) * a`: Integer 0 multiplied by real 1.0 produces real 0.0, then `0.0 * a` produces real 0.0
2. When computing `0 * (1.0 * a)`: Real 1.0 multiplied by symbol a produces a symbolic expression, then `0 * <symbolic>` normalizes to integer 0

The type of the result (INTEGER vs REAL) should not depend on evaluation order. This violates both associativity and type consistency principles. In a symbolic expression system, the type of zero should be consistent regardless of how it was computed.

## Fix

The issue is in the `normalize` function around lines 818-860 in symbolic.py. When normalizing FACTORS expressions that evaluate to zero, the function returns `as_number(coeff)` which creates an INTEGER when coeff is an integer. However, if any operand in the product was REAL, the result should be REAL.

```diff
--- a/symbolic.py
+++ b/symbolic.py
@@ -817,6 +817,7 @@ def normalize(obj):

     if obj.op is Op.FACTORS:
         coeff = 1
+        has_real = False
         d = {}
         for b, e in obj.data.items():
             if e == 0:
@@ -828,6 +829,8 @@ def normalize(obj):

             if b.op in (Op.INTEGER, Op.REAL):
                 if e == 1:
+                    if b.op is Op.REAL:
+                        has_real = True
                     coeff *= b.data[0]
                 elif e > 0:
                     coeff *= b.data[0] ** e
@@ -843,7 +846,11 @@ def normalize(obj):
                 _pairs_add(d, b, e)
         if len(d) == 0 or coeff == 0:
             # TODO: determine correct kind
             assert isinstance(coeff, number_types)
-            return as_number(coeff)
+            if has_real or isinstance(coeff, float):
+                return as_real(float(coeff))
+            else:
+                return as_integer(int(coeff))
```