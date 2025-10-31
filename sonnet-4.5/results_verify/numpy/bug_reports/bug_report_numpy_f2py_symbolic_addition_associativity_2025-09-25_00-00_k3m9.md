# Bug Report: numpy.f2py.symbolic Addition Associativity Violation

**Target**: `numpy.f2py.symbolic.Expr.__add__`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Addition operation violates associativity when operands have different integer kinds, leading to different results based on evaluation order.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from numpy.f2py.symbolic import as_symbol, as_integer, Expr, Op

@st.composite
def expr_integers(draw):
    value = draw(st.integers(min_value=-1000, max_value=1000))
    kind = draw(st.sampled_from([4, 8]))
    return as_integer(value, kind)

@st.composite
def expr_symbols(draw):
    name = draw(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10))
    return as_symbol(name)

@st.composite
def expr_trees(draw, max_depth=3):
    if max_depth == 0:
        choice = draw(st.sampled_from(['int', 'symbol']))
        if choice == 'int':
            return draw(expr_integers())
        else:
            return draw(expr_symbols())

    choice = draw(st.sampled_from(['simple', 'add']))
    if choice == 'simple':
        return draw(expr_trees(max_depth=0))
    else:
        left = draw(expr_trees(max_depth=max_depth-1))
        right = draw(expr_trees(max_depth=max_depth-1))
        return left + right

@given(expr_trees(), expr_trees(), expr_trees())
def test_addition_associative(a, b, c):
    assert (a + b) + c == a + (b + c)
```

**Failing input**: `a=Expr(Op.SYMBOL, 'a')`, `b=Expr(Op.INTEGER, (0, 8))`, `c=Expr(Op.INTEGER, (1, 4))`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages')

from numpy.f2py.symbolic import as_symbol, as_integer

a = as_symbol('a')
b = as_integer(0, 8)
c = as_integer(1, 4)

left_assoc = (a + b) + c
right_assoc = a + (b + c)

print(f"(a + 0_8) + 1_4 = {left_assoc}")
print(f"a + (0_8 + 1_4) = {right_assoc}")
print(f"Equal? {left_assoc == right_assoc}")
```

Output:
```
(a + 0_8) + 1_4 = 1 + a
a + (0_8 + 1_4) = 1_8 + a
Equal? False
```

## Why This Is A Bug

Addition must be associative: `(a + b) + c` should equal `a + (b + c)` for all values. The bug occurs because:

1. When computing `(a + 0_8) + 1_4`: The zero with kind=8 is normalized away first, leaving `a + 1_4`, which preserves kind=4
2. When computing `a + (0_8 + 1_4)`: The integers `0_8 + 1_4` are added first with kind promotion to 8, giving `1_8`, then `a + 1_8` preserves kind=8

This violates the fundamental mathematical property that addition order should not matter. The symbolic expression system incorrectly propagates type information depending on evaluation order.

## Fix

The bug is in the `__add__` method of the `Expr` class around lines 427-453 in symbolic.py. When adding expressions with different kinds, the kind promotion should be determined by the final expression structure, not by intermediate evaluation order.

A potential fix would be to ensure that when normalizing TERMS expressions, the kind is consistently determined by taking the maximum kind among all terms, regardless of the order they were added.

```diff
--- a/symbolic.py
+++ b/symbolic.py
@@ -429,8 +429,12 @@ class Expr:
         other = as_expr(other)
         if isinstance(other, Expr):
             if self.op is other.op:
                 if self.op in (Op.INTEGER, Op.REAL):
+                    # Use max kind for consistent type promotion
+                    kind = max(self.data[1], other.data[1]) if isinstance(self.data[1], int) and isinstance(other.data[1], int) else self.data[1]
                     return as_number(
-                        self.data[0] + other.data[0],
-                        max(self.data[1], other.data[1]))
+                        self.data[0] + other.data[0], kind)
```

However, the real issue is deeper: when a zero term is normalized away in TERMS expressions, the kind information is lost. A more comprehensive fix would ensure that kind promotion is applied consistently across all paths in the normalize function when dealing with TERMS that contain both numeric and symbolic operands.