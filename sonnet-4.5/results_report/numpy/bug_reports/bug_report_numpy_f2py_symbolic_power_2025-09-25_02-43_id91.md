# Bug Report: numpy.f2py.symbolic Power Operator Malformed Output

**Target**: `numpy.f2py.symbolic.Expr.tostring()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `tostring()` method for power expressions produces syntactically invalid output `x * *N` instead of the correct `x**N` for Fortran/Python or `pow(x, N)` for C.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import numpy.f2py.symbolic as sym

simple_expr_chars = st.sampled_from(['x', 'y', 'z', 'a', 'b', 'c'])

@given(st.integers(min_value=1, max_value=10), simple_expr_chars)
@settings(max_examples=200)
def test_power_operator_roundtrip(exp, var):
    expr_str = f'{var}**{exp}'
    e = sym.fromstring(expr_str)
    s = e.tostring()

    assert '**' in s or 'pow' in s.lower() or '^' in s, \
        f"Power operator lost in tostring: {expr_str} -> {s}"

    e2 = sym.fromstring(s)
    assert e == e2, f"Power round-trip failed: {expr_str} -> {s} -> {e2.tostring()}"
```

**Failing input**: `exp=2, var='x'` (i.e., `x**2`)

## Reproducing the Bug

```python
import numpy.f2py.symbolic as sym
from numpy.f2py.symbolic import Language

e = sym.fromstring('x**2')
print(f'Input: x**2')
print(f'Fortran output: {e.tostring(language=Language.Fortran)}')
print(f'Python output: {e.tostring(language=Language.Python)}')
print(f'C output: {e.tostring(language=Language.C)}')
```

Output:
```
Input: x**2
Fortran output: x * *2
Python output: x * *2
C output: x * *2
```

The output `x * *2` is syntactically invalid. It should be:
- Fortran: `x**2`
- Python: `x**2`
- C: `pow(x, 2)`

## Why This Is A Bug

The power operator `**` is a fundamental arithmetic operation in Fortran and Python. The `tostring()` method should produce valid, syntactically correct code in the target language.

The string `x * *2` is not valid syntax in any of these languages:
- In Fortran/Python, it would be interpreted as `x * (*2)` (multiplication by a dereferenced pointer), which is nonsensical
- The code cannot be used as-is in generated Fortran code
- While the expression accidentally round-trips through `fromstring()` (because the parser somehow accepts this malformed syntax), this is fragile and produces incorrect code for any downstream use

## Fix

The `tostring()` method needs to handle the power operation (represented internally with `Op.DEREF`) correctly:
- For Fortran and Python: emit `base**exponent`
- For C: emit `pow(base, exponent)`

The fix would be in the code that handles `Op.DEREF` or `Op.FACTORS` with exponent data in `Expr.tostring()`.