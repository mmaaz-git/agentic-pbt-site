# Bug Report: numpy.f2py.symbolic.Expr.parse Power Operator Parsing

**Target**: `numpy.f2py.symbolic.Expr.parse`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Expr.parse()` method incorrectly parses the Fortran power operator `**`, treating it as two separate operators: multiplication `*` followed by dereference `*`. This breaks the fundamental round-trip property and causes incorrect expression parsing.

## Property-Based Test

```python
import numpy.f2py.symbolic as symbolic
from hypothesis import given, strategies as st, settings, assume


simple_fortran_expr = st.one_of(
    st.just("x"),
    st.just("y"),
    st.just("a"),
    st.just("b"),
    st.integers(min_value=-1000, max_value=1000).map(str),
)


@st.composite
def fortran_expr(draw):
    size = draw(st.integers(min_value=0, max_value=3))

    if size == 0:
        return draw(simple_fortran_expr)

    op = draw(st.sampled_from(['+', '-', '*', '**']))

    left = draw(fortran_expr())
    right = draw(simple_fortran_expr)

    if op == '**':
        return f"({left}) ** ({right})"
    else:
        return f"{left} {op} {right}"


@settings(max_examples=200)
@given(fortran_expr())
def test_parse_tostring_roundtrip(expr_str):
    try:
        parsed = symbolic.Expr.parse(expr_str)
    except Exception:
        assume(False)

    stringified = parsed.tostring()

    try:
        reparsed = symbolic.Expr.parse(stringified)
    except Exception as e:
        raise AssertionError(
            f"Round-trip failed: parse('{expr_str}') -> tostring() -> '{stringified}' "
            f"-> parse failed with {e}"
        )

    assert reparsed == parsed, (
        f"Round-trip not equal: parse('{expr_str}') = {parsed}, "
        f"but parse(tostring()) = {reparsed}"
    )
```

**Failing input**: `'x * x'`

## Reproducing the Bug

```python
import numpy.f2py.symbolic as symbolic

original = symbolic.Expr.parse('x * x')
print(f"parse('x * x') = {original}")
print(f"repr: {repr(original)}")

output = original.tostring()
print(f"tostring() = '{output}'")

reparsed = symbolic.Expr.parse(output)
print(f"parse('{output}') = {reparsed}")
print(f"repr: {repr(reparsed)}")

print(f"\nEqual? {original == reparsed}")
```

**Output:**
```
parse('x * x') = x ** 2
repr: Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'x'): 2})
tostring() = 'x ** 2'
parse('x ** 2') = x * *2
repr: Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'x'): 1, Expr(Op.DEREF, Expr(Op.INTEGER, (2, 4))): 1})

Equal? False
```

## Why This Is A Bug

1. **Violates round-trip property**: The fundamental invariant `parse(e.tostring()) == e` is broken
2. **Incorrect semantics**: `x ** 2` (x to the power of 2) is being parsed as `x * (*2)` (x multiplied by dereference of 2)
3. **Fortran incompatibility**: The `**` operator is standard in Fortran for exponentiation, but the parser doesn't recognize it
4. **Internal inconsistency**: The library can *generate* `x ** 2` from `x * x` but cannot *parse* `x ** 2` correctly

This is a high-severity logic bug because:
- It affects core parsing functionality
- It silently produces incorrect results
- It breaks a fundamental property that users would reasonably expect

## Fix

The parser needs to recognize `**` as a single token for the power operator, not as two separate `*` tokens. The tokenization logic should match `**` before attempting to match individual `*` operators.

The fix would involve modifying the tokenization or parsing logic in `symbolic.py` to handle `**` as a distinct operator token, similar to how other two-character operators (like `>=`, `<=`) are likely handled.