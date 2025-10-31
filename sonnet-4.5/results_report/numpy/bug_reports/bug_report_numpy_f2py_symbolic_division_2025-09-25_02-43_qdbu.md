# Bug Report: numpy.f2py.symbolic Division Round-Trip Failure

**Target**: `numpy.f2py.symbolic.Expr.tostring()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `tostring()` method for division expressions produces `ArithOp.DIV(x, y)` syntax for Fortran and Python languages, which cannot be parsed back correctly, breaking the fundamental round-trip property.

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
```

**Failing input**: `'(x) / (y)'`

## Reproducing the Bug

```python
import numpy.f2py.symbolic as sym
from numpy.f2py.symbolic import Language

e1 = sym.fromstring('x / y')
print(f'Original expression: {e1}')
print(f'tostring() output: {e1.tostring()}')

e2 = sym.fromstring(e1.tostring())
print(f'Re-parsed expression: {e2}')
print(f'Are they equal? {e1 == e2}')
print(f'Type of e1.data[0]: {type(e1.data[0])}')
print(f'Type of e2.data[0]: {type(e2.data[0])}')
```

Output:
```
Original expression: ArithOp.DIV(x, y)
tostring() output: ArithOp.DIV(x, y)
Re-parsed expression: ArithOp.DIV(a, b)
Are they equal? False
Type of e1.data[0]: <enum 'ArithOp'>
Type of e2.data[0]: <class 'numpy.f2py.symbolic.Expr'>
```

The warning message confirms the issue: `fromstring: treating 'ArithOp.DIV' as symbol`

## Why This Is A Bug

The `tostring()` method is supposed to generate valid Fortran/Python syntax that can be parsed back. When a division expression is converted to string, it should produce `x / y` (which can be re-parsed), not `ArithOp.DIV(x, y)` (which gets interpreted as a function call to a symbol named 'ArithOp.DIV').

This breaks the fundamental round-trip property: `fromstring(expr.tostring()) == expr`

The bug only affects Fortran and Python language modes. The C language mode correctly produces `x / y`.

## Fix

The `tostring()` method should check the language mode and produce appropriate division syntax:
- Fortran: `x / y`
- Python: `x / y`
- C: `x / y` (already works correctly)

The fix would be in the `Expr.tostring()` method to handle `Op.APPLY` with `ArithOp.DIV` specially and emit `/` operator syntax instead of the enum representation.