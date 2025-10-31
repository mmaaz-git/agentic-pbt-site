# Bug Report: numpy.f2py.symbolic.Expr.parse ZeroDivisionError

**Target**: `numpy.f2py.symbolic.Expr.parse`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Expr.parse` method crashes with `ZeroDivisionError` when parsing float division by zero expressions (e.g., `'1.5 / 0'`), even though division by zero is valid in symbolic expressions and integer division by zero is handled correctly.

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

**Failing input**: `'1.5 / 0'`

## Reproducing the Bug

```python
import numpy.f2py.symbolic as symbolic

expr = symbolic.Expr.parse('1.5 / 0')
```

Output:
```
Traceback (most recent call last):
  File "test.py", line 3, in <module>
    expr = symbolic.Expr.parse('1.5 / 0')
  File ".../numpy/f2py/symbolic.py", line 167, in parse
    return fromstring(s, language=language)
  File ".../numpy/f2py/symbolic.py", line 1276, in fromstring
    r = _FromStringWorker(language=language).parse(s)
  File ".../numpy/f2py/symbolic.py", line 1314, in parse
    return self.process(unquoted)
  File ".../numpy/f2py/symbolic.py", line 1423, in process
    result /= operand
  File ".../numpy/f2py/symbolic.py", line 542, in __truediv__
    return normalize(as_apply(ArithOp.DIV, self, other))
  File ".../numpy/f2py/symbolic.py", line 870, in normalize
    c1, c2 = c1 / c2, 1
ZeroDivisionError: float division by zero
```

## Why This Is A Bug

1. The parser should handle division by zero gracefully since it's representing symbolic expressions, not computing them
2. Integer division by zero (`'1 / 0'`) parses successfully, but float division by zero crashes inconsistently
3. The crash occurs in the normalization step at line 870, where it attempts to actually perform float division instead of keeping it symbolic

## Fix

The `normalize` function in `symbolic.py` around line 870 should check for zero divisors before attempting division:

```diff
--- a/numpy/f2py/symbolic.py
+++ b/numpy/f2py/symbolic.py
@@ -867,7 +867,7 @@ def normalize(obj):
                             # In the following, we assume that n/d is
                             # a constant expression
                             assert isinstance(c1, number_types)
-                            c1, c2 = c1 / c2, 1
+                            c1, c2 = (c1 / c2 if c2 != 0 else c1, 1 if c2 != 0 else c2)
                             numer_denom = (c1 * n, d)
                         else:
                             numer_denom = (c1, c2)
```