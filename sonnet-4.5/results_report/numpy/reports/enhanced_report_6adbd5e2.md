# Bug Report: numpy.f2py.symbolic Addition Associativity Violation

**Target**: `numpy.f2py.symbolic.Expr.__add__`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The addition operation in numpy.f2py.symbolic violates associativity when operands have different integer kinds, producing different results based on evaluation order.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages')

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
    left_assoc = (a + b) + c
    right_assoc = a + (b + c)
    assert left_assoc == right_assoc, f"Associativity violated: ({a} + {b}) + {c} = {left_assoc} != {right_assoc} = {a} + ({b} + {c})"

if __name__ == "__main__":
    test_addition_associative()
```

<details>

<summary>
**Failing input**: `a=Expr(Op.INTEGER, (0, 8)), b=Expr(Op.INTEGER, (1, 4)), c=Expr(Op.SYMBOL, 'a')`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 42, in <module>
    test_addition_associative()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 36, in test_addition_associative
    def test_addition_associative(a, b, c):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 39, in test_addition_associative
    assert left_assoc == right_assoc, f"Associativity violated: ({a} + {b}) + {c} = {left_assoc} != {right_assoc} = {a} + ({b} + {c})"
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Associativity violated: (0_8 + 1) + a = 1_8 + a != 1 + a = 0_8 + (1 + a)
Falsifying example: test_addition_associative(
    a=Expr(Op.INTEGER, (0, 8)),
    b=Expr(Op.INTEGER, (1, 4)),
    c=Expr(Op.SYMBOL, 'a'),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/f2py/symbolic.py:250
        /home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/f2py/symbolic.py:274
        /home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/f2py/symbolic.py:280
        /home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/f2py/symbolic.py:318
        /home/npc/miniconda/lib/python3.13/enum.py:1336
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages')

from numpy.f2py.symbolic import as_symbol, as_integer

# Create expressions
a = as_symbol('a')
b = as_integer(0, 8)  # 0 with kind=8
c = as_integer(1, 4)  # 1 with kind=4

# Test associativity: (a + b) + c vs a + (b + c)
left_assoc = (a + b) + c
right_assoc = a + (b + c)

print("Testing addition associativity violation:")
print(f"a = {a}")
print(f"b = {b} (0 with kind=8)")
print(f"c = {c} (1 with kind=4)")
print()
print(f"(a + 0_8) + 1_4 = {left_assoc}")
print(f"a + (0_8 + 1_4) = {right_assoc}")
print()
print(f"Are they equal? {left_assoc == right_assoc}")
print(f"Should be equal for associativity to hold: True")
print()
print("This violates the fundamental property of associativity: (a + b) + c == a + (b + c)")
```

<details>

<summary>
Associativity violation: left-associative vs right-associative addition produces different results
</summary>
```
Testing addition associativity violation:
a = a
b = 0_8 (0 with kind=8)
c = 1 (1 with kind=4)

(a + 0_8) + 1_4 = 1 + a
a + (0_8 + 1_4) = 1_8 + a

Are they equal? False
Should be equal for associativity to hold: True

This violates the fundamental property of associativity: (a + b) + c == a + (b + c)
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical property of associativity that states `(a + b) + c` must equal `a + (b + c)` for all values. The bug occurs because the normalization process in numpy.f2py.symbolic incorrectly handles integer kinds when zeros are involved:

1. **Left-associative evaluation `(a + 0_8) + 1_4`**:
   - First: `a + 0_8` normalizes to just `a` (zero is removed, kind=8 information lost)
   - Then: `a + 1_4` results in `1 + a` where 1 has kind=4

2. **Right-associative evaluation `a + (0_8 + 1_4)`**:
   - First: `0_8 + 1_4` becomes `1_8` (kind is promoted to max(8,4)=8)
   - Then: `a + 1_8` results in `1_8 + a` where 1 has kind=8

The result depends on evaluation order, which is mathematically incorrect. This affects F2PY's ability to correctly process Fortran expressions with mixed precision, potentially leading to incorrect dimension specifications and type mismatches.

## Relevant Context

The numpy.f2py.symbolic module is critical for F2PY (Fortran to Python interface generator) which parses and evaluates Fortran expressions. The module implements a symbolic engine for analyzing dimension specifications and other expressions in Fortran code. The bug is in the interaction between:

1. The `__add__` method (lines 427-453 in symbolic.py) which handles addition with kind promotion
2. The `normalize` function (lines 790-924) which simplifies expressions but loses kind information when zeros are removed
3. Line 810 contains a TODO comment: `# TODO: determine correct kind` indicating this is a known area needing improvement

Key source locations:
- `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/f2py/symbolic.py:427-453` (__add__ method)
- `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/f2py/symbolic.py:810` (TODO comment about kind determination)

## Proposed Fix

The issue stems from the normalize function returning a zero without preserving the maximum kind from the original terms. When all terms cancel out to zero, the function should track and preserve the maximum kind from all terms that were present.

```diff
--- a/symbolic.py
+++ b/symbolic.py
@@ -796,6 +796,7 @@ def normalize(obj):
     if obj.op is Op.TERMS:
         d = {}
+        max_kind = 4  # Track maximum kind seen
         for t, c in obj.data.items():
             if c == 0:
                 continue
@@ -804,11 +805,16 @@ def normalize(obj):
                 c = 1
             if t.op is Op.TERMS:
                 for t1, c1 in t.data.items():
+                    if t1.op in (Op.INTEGER, Op.REAL) and isinstance(t1.data[1], int):
+                        max_kind = max(max_kind, t1.data[1])
                     _pairs_add(d, t1, c1 * c)
             else:
+                if t.op in (Op.INTEGER, Op.REAL) and isinstance(t.data[1], int):
+                    max_kind = max(max_kind, t.data[1])
                 _pairs_add(d, t, c)
         if len(d) == 0:
-            # TODO: determine correct kind
-            return as_number(0)
+            # Return zero with the maximum kind seen in the terms
+            return as_number(0, max_kind)
         elif len(d) == 1:
             (t, c), = d.items()
             if c == 1:
```