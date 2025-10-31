# Bug Report: numpy.f2py.symbolic Negative Number Addition Parsing

**Target**: `numpy.f2py.symbolic.fromstring`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The parser incorrectly handles expressions like `-1 + -1`, creating a phantom empty symbol instead of correctly evaluating to `-2`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy.f2py.symbolic as sym

identifiers = st.text(
    alphabet=st.characters(whitelist_categories=('Ll', 'Lu'), min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=10
).filter(lambda s: s and s[0].isalpha())

@given(identifiers, st.integers(min_value=-100, max_value=100))
@settings(max_examples=300)
def test_substitute_with_number(var_name, num):
    try:
        expr = sym.fromstring(f"{var_name} + {var_name}")
        var = sym.fromstring(var_name)
        num_expr = sym.as_number(num)
    except (ValueError, KeyError, RecursionError):
        assume(False)

    result = expr.substitute({var: num_expr})
    expected = sym.fromstring(f"{num} + {num}")

    assert sym.normalize(result) == sym.normalize(expected), \
        f"Substituting {var_name} with {num} in {expr} failed"
```

**Failing input**: `var_name='a', num=-1`

## Reproducing the Bug

```python
import numpy.f2py.symbolic as sym

expr = sym.fromstring("-1 + -1")
print(f"Parsed: {repr(expr)}")
print(f"String: {expr}")

expected = sym.fromstring("-2")
print(f"\nExpected: {repr(expected)}")
print(f"Equal: {expr == expected}")
```

Output:
```
Parsed: Expr(Op.TERMS, {Expr(Op.INTEGER, (1, 4)): -2, Expr(Op.SYMBOL, ''): 1})
String: -2 +
Expected: Expr(Op.INTEGER, (-2, 4))
Equal: False
```

## Why This Is A Bug

When parsing `-1 + -1`, the parser should recognize this as the sum of two negative integers and evaluate to `-2`. Instead, it creates an expression containing both `-2` and an empty symbol `''`, representing `-2 * 1 + 1 * ''`.

The warning message confirms this: `ExprWarning: fromstring: treating '' as symbol (original=-1 + -1)`.

This causes incorrect behavior when:
1. Parsing expressions with consecutive negative signs after operators (`+ -`, `- -`)
2. Substituting variables with negative numbers
3. Any operation that produces strings like `x + -y`

The workaround is to use parentheses: `(-1) + (-1)` parses correctly.

## Fix

The parser's tokenizer needs to correctly handle the sequence `+ -` by recognizing that `-` after a binary operator starts a negative number literal, not a separate operation that leaves an empty operand. The issue likely lies in how the `_FromStringWorker` class tokenizes the input string when it encounters `+ -` or `- -` sequences.