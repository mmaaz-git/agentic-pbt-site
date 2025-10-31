# Bug Report: numpy.f2py.symbolic Division Expression Round-Trip Failure

**Target**: `numpy.f2py.symbolic.Expr.tostring()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `tostring()` method in numpy.f2py.symbolic produces `ArithOp.DIV(x, y)` syntax for division expressions in Fortran and Python language modes, which cannot be parsed back correctly, breaking the fundamental round-trip property that `fromstring(expr.tostring()) == expr`.

## Property-Based Test

```python
from hypothesis import assume, given, settings, strategies as st
import numpy.f2py.symbolic as sym

simple_expr_chars = st.sampled_from(['x', 'y', 'z', 'a', 'b', 'c'])

@st.composite
def simple_fortran_expr(draw):
    choice = draw(st.integers(min_value=0, max_value=4))
    if choice == 0:
        return draw(simple_expr_chars)
    elif choice == 1:
        return str(draw(st.integers(min_value=-100, max_value=100)))
    elif choice == 2:
        left = draw(simple_fortran_expr())
        right = draw(simple_fortran_expr())
        op = draw(st.sampled_from(['+', '-', '*']))
        return f'({left} {op} {right})'
    elif choice == 3:
        left = draw(simple_fortran_expr())
        right = draw(simple_fortran_expr())
        return f'({left}) / ({right})'
    else:
        base = draw(simple_expr_chars)
        exp = draw(st.integers(min_value=1, max_value=5))
        return f'{base}**{exp}'

@given(simple_fortran_expr())
@settings(max_examples=500)
def test_fromstring_tostring_roundtrip(expr_str):
    try:
        e1 = sym.fromstring(expr_str)
        s = e1.tostring()
        e2 = sym.fromstring(s)
        assert e1 == e2, f"Round-trip failed: {expr_str} -> {s} -> {e2.tostring()}"
    except Exception as ex:
        if "division by zero" in str(ex).lower() or "zerodivision" in str(ex).lower():
            assume(False)
        raise

if __name__ == "__main__":
    test_fromstring_tostring_roundtrip()
```

<details>

<summary>
**Failing input**: `'(x) / (y)'`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:1514: ExprWarning: fromstring: treating 'ArithOp.DIV' as symbol (original=ArithOp.DIV(-44, 2 * b * z * *2 * *4))
  ewarn(
/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:1514: ExprWarning: fromstring: treating 'ArithOp.DIV' as symbol (original=ArithOp.DIV(-924, 2 * b * z * *2 * *4))
  ewarn(
/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:1514: ExprWarning: fromstring: treating 'ArithOp.DIV' as symbol (original=ArithOp.DIV(-924, 2 * z ** 2 * *2 * *4))
  ewarn(
/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:1514: ExprWarning: fromstring: treating 'ArithOp.DIV' as symbol (original=ArithOp.DIV(-924, 2 * (-44) ** -1 * z ** 2 * (*2) ** 2))
  ewarn(
/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:1514: ExprWarning: fromstring: treating '' as symbol (original=ArithOp.DIV(-924, 2 * (-44) ** -1 * z ** 2 * (*2) ** 2))
  ewarn(
/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:1514: ExprWarning: fromstring: treating 'ArithOp.DIV' as symbol (original=ArithOp.DIV(-44, (-44) ** -1 * z ** 2 * (*2) ** 2))
  ewarn(
/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:1514: ExprWarning: fromstring: treating '' as symbol (original=ArithOp.DIV(-44, (-44) ** -1 * z ** 2 * (*2) ** 2))
  ewarn(
/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:1514: ExprWarning: fromstring: treating 'ArithOp.DIV' as symbol (original=ArithOp.DIV(-1848, (-44) ** -1 * x * z ** 2 * (*2) ** 2))
  ewarn(
/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:1514: ExprWarning: fromstring: treating '' as symbol (original=ArithOp.DIV(-1848, (-44) ** -1 * x * z ** 2 * (*2) ** 2))
  ewarn(
/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:1514: ExprWarning: fromstring: treating '' as symbol (original=c * *4 - 11 * (b * *5) ** -1)
  ewarn(
/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:1514: ExprWarning: fromstring: treating 'ArithOp.DIV' as symbol (original=ArithOp.DIV(a, 42))
  ewarn(
/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:1514: ExprWarning: fromstring: treating 'ArithOp.DIV' as symbol (original=ArithOp.DIV(x, 42))
  ewarn(
/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:1514: ExprWarning: fromstring: treating 'ArithOp.DIV' as symbol (original=ArithOp.DIV(x, b))
  ewarn(
/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:1514: ExprWarning: fromstring: treating 'ArithOp.DIV' as symbol (original=ArithOp.DIV(x, y))
  ewarn(
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 41, in <module>
    test_fromstring_tostring_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 28, in test_fromstring_tostring_roundtrip
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 34, in test_fromstring_tostring_roundtrip
    assert e1 == e2, f"Round-trip failed: {expr_str} -> {s} -> {e2.tostring()}"
           ^^^^^^^^
AssertionError: Round-trip failed: (x) / (y) -> ArithOp.DIV(x, y) -> ArithOp.DIV(x, y)
Falsifying example: test_fromstring_tostring_roundtrip(
    expr_str='(x) / (y)',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/38/hypo.py:19
        /home/npc/pbt/agentic-pbt/worker_/38/hypo.py:20
        /home/npc/pbt/agentic-pbt/worker_/38/hypo.py:21
        /home/npc/pbt/agentic-pbt/worker_/38/hypo.py:35
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:209
        (and 18 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
import numpy.f2py.symbolic as sym
from numpy.f2py.symbolic import Language

# Demonstrate the division round-trip bug
e1 = sym.fromstring('x / y')
print(f'Original expression: {e1}')
print(f'Type of e1: {type(e1)}')
print(f'e1.op: {e1.op}')
print(f'e1.data: {e1.data}')
print()

# Show tostring output for different languages
print('tostring() outputs by language:')
print(f'  Fortran: {e1.tostring(language=Language.Fortran)}')
print(f'  Python:  {e1.tostring(language=Language.Python)}')
print(f'  C:       {e1.tostring(language=Language.C)}')
print()

# Try to parse back the Fortran output
fortran_string = e1.tostring(language=Language.Fortran)
print(f'Attempting to re-parse Fortran output: "{fortran_string}"')
try:
    e2 = sym.fromstring(fortran_string)
    print(f'Re-parsed expression: {e2}')
    print(f'Type of e2: {type(e2)}')
    print(f'e2.op: {e2.op}')
    print(f'e2.data: {e2.data}')
    print(f'Are they equal? {e1 == e2}')
    if e1 != e2:
        print(f'FAILURE: Round-trip failed!')
        print(f'  e1.data[0]: {e1.data[0]}, type: {type(e1.data[0])}')
        print(f'  e2.data[0]: {e2.data[0]}, type: {type(e2.data[0])}')
except Exception as ex:
    print(f'ERROR during re-parsing: {ex}')
print()

# Try the same with Python language mode
python_string = e1.tostring(language=Language.Python)
print(f'Attempting to re-parse Python output: "{python_string}"')
try:
    e3 = sym.fromstring(python_string, language=Language.Python)
    print(f'Re-parsed expression: {e3}')
    print(f'Are they equal? {e1 == e3}')
    if e1 != e3:
        print(f'FAILURE: Round-trip failed for Python mode!')
except Exception as ex:
    print(f'ERROR during re-parsing: {ex}')
print()

# Try with C mode
c_string = e1.tostring(language=Language.C)
print(f'Attempting to re-parse C output: "{c_string}"')
try:
    e4 = sym.fromstring(c_string, language=Language.C)
    print(f'Re-parsed expression: {e4}')
    print(f'Are they equal? {e1 == e4}')
    if e1 == e4:
        print(f'SUCCESS: Round-trip works in C mode!')
except Exception as ex:
    print(f'ERROR during re-parsing: {ex}')
```

<details>

<summary>
Output demonstrating the bug
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:1514: ExprWarning: fromstring: treating 'ArithOp.DIV' as symbol (original=ArithOp.DIV(x, y))
  ewarn(
Original expression: ArithOp.DIV(x, y)
Type of e1: <class 'numpy.f2py.symbolic.Expr'>
e1.op: Op.APPLY
e1.data: (<ArithOp.DIV: 6>, (Expr(Op.SYMBOL, 'x'), Expr(Op.SYMBOL, 'y')), {})

tostring() outputs by language:
  Fortran: ArithOp.DIV(x, y)
  Python:  ArithOp.DIV(x, y)
  C:       x / y

Attempting to re-parse Fortran output: "ArithOp.DIV(x, y)"
Re-parsed expression: ArithOp.DIV(x, y)
Type of e2: <class 'numpy.f2py.symbolic.Expr'>
e2.op: Op.APPLY
e2.data: (Expr(Op.SYMBOL, 'ArithOp.DIV'), (Expr(Op.SYMBOL, 'x'), Expr(Op.SYMBOL, 'y')), {})
Are they equal? False
FAILURE: Round-trip failed!
  e1.data[0]: ArithOp.DIV, type: <enum 'ArithOp'>
  e2.data[0]: ArithOp.DIV, type: <class 'numpy.f2py.symbolic.Expr'>

Attempting to re-parse Python output: "ArithOp.DIV(x, y)"
Re-parsed expression: ArithOp.DIV(x, y)
Are they equal? False
FAILURE: Round-trip failed for Python mode!

Attempting to re-parse C output: "x / y"
Re-parsed expression: ArithOp.DIV(x, y)
Are they equal? True
SUCCESS: Round-trip works in C mode!
```
</details>

## Why This Is A Bug

The `tostring()` method is designed to generate parseable string representations of expressions that can be converted back to the same expression object via `fromstring()`. This round-trip property is essential for serialization, debugging, and code generation.

When a division expression (`x / y`) is parsed, it creates an `Expr` object with `Op.APPLY` operation and `ArithOp.DIV` enum as the function. However, when `tostring()` is called with Fortran or Python language modes, it outputs `ArithOp.DIV(x, y)` instead of the expected `x / y` syntax.

When this output is parsed back using `fromstring()`, the parser treats `ArithOp.DIV` as a symbol name (as evidenced by the warning message) rather than recognizing it as the division operation enum. This creates a fundamentally different expression object where:
- The original has `data[0] = ArithOp.DIV` (an enum value)
- The re-parsed version has `data[0] = Expr(Op.SYMBOL, 'ArithOp.DIV')` (a symbol expression)

This breaks the equality check and violates the expected invariant that parsing the string representation of an expression should yield an equivalent expression.

## Relevant Context

The code at lines 361-366 of `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/f2py/symbolic.py` shows that special handling for division already exists for C language mode:

```python
if name is ArithOp.DIV and language is Language.C:
    numer, denom = [arg.tostring(Precedence.PRODUCT,
                                 language=language)
                    for arg in args]
    r = f'{numer} / {denom}'
    precedence = Precedence.PRODUCT
```

This special case correctly outputs `x / y` for C mode, which is why the round-trip works in C mode. The same treatment needs to be applied to Fortran and Python language modes.

The parser at lines 1405-1424 correctly handles the `/` operator for all language modes, creating the proper division expression. The asymmetry is entirely in the `tostring()` method.

## Proposed Fix

Extend the special division handling in the `tostring()` method to cover all language modes, not just C:

```diff
--- a/numpy/f2py/symbolic.py
+++ b/numpy/f2py/symbolic.py
@@ -358,11 +358,11 @@ class Expr:
             precedence = Precedence.PRODUCT if factors else Precedence.ATOM
         elif self.op is Op.APPLY:
             name, args, kwargs = self.data
-            if name is ArithOp.DIV and language is Language.C:
+            if name is ArithOp.DIV:
                 numer, denom = [arg.tostring(Precedence.PRODUCT,
                                              language=language)
                                 for arg in args]
                 r = f'{numer} / {denom}'
                 precedence = Precedence.PRODUCT
             else:
                 args = [arg.tostring(Precedence.TUPLE, language=language)
```