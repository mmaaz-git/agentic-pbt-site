# Bug Report: numpy.f2py.symbolic Power Operator Round-Trip Failure

**Target**: `numpy.f2py.symbolic.fromstring` / `Expr.tostring`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The power operator (`**`) is correctly generated in `Expr.tostring()` output but incorrectly parsed by `fromstring()`, breaking the round-trip property for expressions with exponentiation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy.f2py.symbolic as sym

identifiers = st.text(
    alphabet=st.characters(whitelist_categories=('Ll', 'Lu'), min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=10
).filter(lambda s: s and s[0].isalpha())

simple_exprs = st.recursive(
    identifiers | st.integers(min_value=-100, max_value=100).map(str),
    lambda children: st.one_of(
        st.builds(lambda a, b: f"({a} + {b})", children, children),
        st.builds(lambda a, b: f"({a} - {b})", children, children),
        st.builds(lambda a, b: f"({a} * {b})", children, children),
    ),
    max_leaves=5,
)

@given(simple_exprs)
@settings(max_examples=500)
def test_fromstring_str_roundtrip(expr_str):
    try:
        expr = sym.fromstring(expr_str)
    except (ValueError, KeyError, RecursionError) as e:
        assume(False)

    str_repr = str(expr)

    try:
        expr2 = sym.fromstring(str_repr)
    except (ValueError, KeyError, RecursionError) as e:
        assert False, f"Failed to parse back string representation: {str_repr!r}, error: {e}"

    assert expr == expr2, f"Round-trip failed: {expr_str!r} -> {expr!r} -> {str_repr!r} -> {expr2!r}"
```

**Failing input**: `'(a * a)'`

## Reproducing the Bug

```python
import numpy.f2py.symbolic as sym

expr1 = sym.fromstring("a * a")
print(f"Step 1 - Parse 'a * a': {repr(expr1)}")

str_repr = str(expr1)
print(f"Step 2 - Convert to string: '{str_repr}'")

expr2 = sym.fromstring(str_repr)
print(f"Step 3 - Parse '{str_repr}': {repr(expr2)}")

print(f"\nAre they equal? {expr1 == expr2}")
```

Output:
```
Step 1 - Parse 'a * a': Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'a'): 2})
Step 2 - Convert to string: 'a ** 2'
Step 3 - Parse 'a ** 2': Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'a'): 1, Expr(Op.DEREF, Expr(Op.INTEGER, (2, 4))): 1})

Are they equal? False
```

## Why This Is A Bug

The bug violates the fundamental round-trip property: `fromstring(str(expr)) == expr`.

When `a * a` is parsed, it's correctly normalized to `a ** 2` (base `a` with exponent 2). The `tostring()` method outputs this as the string `"a ** 2"`. However, `fromstring()` parses `**` as `* *` (multiplication followed by dereference operator), treating it as `a * (*2)` instead of exponentiation.

This inconsistency means that expressions with exponents cannot be reliably serialized and deserialized, breaking the contract between `fromstring()` and `tostring()`.

## Fix

The parser needs to recognize `**` as an exponentiation operator. The fix requires updating the tokenizer in the `_FromStringWorker` class to handle `**` as a single token representing exponentiation, rather than treating it as two separate `*` tokens. The parser should recognize `**` as a power operator with appropriate precedence, similar to how Python and Fortran handle exponentiation.