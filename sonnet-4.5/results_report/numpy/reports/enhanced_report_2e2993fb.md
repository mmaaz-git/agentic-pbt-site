# Bug Report: numpy.f2py.symbolic.Expr - Parse/Tostring Round-trip Failure for Exponentiation

**Target**: `numpy.f2py.symbolic.Expr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Expr.parse()` function incorrectly parses the exponentiation operator `**` as multiplication followed by a dereference operator when using the default C language mode, breaking the round-trip property between `tostring()` and `parse()`.

## Property-Based Test

```python
#!/usr/bin/env python3

import numpy.f2py.symbolic as symbolic
from hypothesis import given, strategies as st, settings

st_symbol_name = st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz')

@st.composite
def st_simple_expr(draw):
    choice = draw(st.integers(min_value=0, max_value=2))
    if choice == 0:
        return symbolic.as_number(draw(st.integers(min_value=-1000, max_value=1000)))
    elif choice == 1:
        return symbolic.as_symbol(draw(st_symbol_name))
    else:
        left = symbolic.as_symbol(draw(st_symbol_name))
        right = symbolic.as_symbol(draw(st_symbol_name))
        return left * right

@given(st_simple_expr())
@settings(max_examples=500)
def test_expr_parse_tostring_roundtrip(expr):
    s = expr.tostring()
    parsed = symbolic.Expr.parse(s)
    assert parsed == expr, f"Round-trip failed for {repr(expr)}: tostring() = '{s}', parsed = {repr(parsed)}"

if __name__ == "__main__":
    test_expr_parse_tostring_roundtrip()
```

<details>

<summary>
**Failing input**: `Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'a'): 2})`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 28, in <module>
    test_expr_parse_tostring_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 21, in test_expr_parse_tostring_roundtrip
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 25, in test_expr_parse_tostring_roundtrip
    assert parsed == expr, f"Round-trip failed for {repr(expr)}: tostring() = '{s}', parsed = {repr(parsed)}"
           ^^^^^^^^^^^^^^
AssertionError: Round-trip failed for Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'a'): 2}): tostring() = 'a ** 2', parsed = Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'a'): 1, Expr(Op.DEREF, Expr(Op.INTEGER, (2, 4))): 1})
Falsifying example: test_expr_parse_tostring_roundtrip(
    expr=Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'a'): 2}),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:142
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:206
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:271
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:333
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:349
        (and 4 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3

import numpy.f2py.symbolic as symbolic

# Create an expression that is a * a (which should be equivalent to a^2)
expr = symbolic.as_symbol('a') * symbolic.as_symbol('a')
print(f'Original expression: {repr(expr)}')

# Convert to string - this should produce "a ** 2"
s = expr.tostring()
print(f'tostring() output: "{s}"')

# Parse the string back
parsed = symbolic.Expr.parse(s)
print(f'Parsed expression: {repr(parsed)}')

# Check if round-trip is successful
print(f'Round-trip successful: {parsed == expr}')
```

<details>

<summary>
Parsed expression contains DEREF instead of exponent
</summary>
```
Original expression: Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'a'): 2})
tostring() output: "a ** 2"
Parsed expression: Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'a'): 1, Expr(Op.DEREF, Expr(Op.INTEGER, (2, 4))): 1})
Round-trip successful: False
```
</details>

## Why This Is A Bug

The symbolic expression module is designed to parse and serialize Fortran/C expressions, maintaining semantic equivalence through round-trip operations. When an expression `a * a` is converted to its string representation, it correctly produces `"a ** 2"` using the exponentiation operator. However, when parsing this string back, the parser incorrectly interprets `**` as two separate operators:

1. The first `*` is treated as multiplication
2. The second `*` is treated as the C dereference operator

This results in the expression being parsed as `a * (*2)` (multiply `a` by dereferenced `2`) instead of `a ** 2` (a squared). This violates the fundamental contract that `parse(expr.tostring()) == expr`, which is essential for:
- Serialization and deserialization of symbolic expressions
- Storing and retrieving expressions from files or databases
- Communication between different parts of the f2py system

The bug occurs because the parser's multiplication/division regex pattern in line 1406 of symbolic.py splits on single `*` characters without accounting for the `**` operator when in C language mode (the default).

## Relevant Context

The issue is in the `_FromStringWorker.process()` method in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py`. When the language is C (default), the code at lines 1405-1424 handles multiplication/division operations:

```python
# multiplication/division operations
operands = re.split(r'(?<=[@\w\d_])\s*([*]|/)',
                    (r if self.language is Language.C
                     else r.replace('**', '@__f2py_DOUBLE_STAR@')))
```

When `language is Language.C`, the string `"a ** 2"` is not protected from being split on `*`, resulting in `['a ', '*', '', '*', ' 2']`. The empty string between the two asterisks combined with the dereference handling at lines 1426-1430 causes the misinterpretation.

NumPy f2py documentation: https://numpy.org/doc/stable/f2py/
Source code: https://github.com/numpy/numpy/blob/main/numpy/f2py/symbolic.py

## Proposed Fix

The parser needs to handle `**` as a single token even in C language mode when it appears in the output of `tostring()`. Since `tostring()` uses `**` for exponentiation regardless of language, the parser should recognize this pattern. Here's a potential fix:

```diff
--- a/numpy/f2py/symbolic.py
+++ b/numpy/f2py/symbolic.py
@@ -1403,10 +1403,13 @@ class _FromStringWorker:
                         tuple(self.process(operands)))

         # multiplication/division operations
-        operands = re.split(r'(?<=[@\w\d_])\s*([*]|/)',
-                            (r if self.language is Language.C
-                             else r.replace('**', '@__f2py_DOUBLE_STAR@')))
+        # Always protect ** from being split, as tostring() uses it for exponentiation
+        # regardless of the language setting
+        r_protected = r.replace('**', '@__f2py_DOUBLE_STAR@')
+        operands = re.split(r'(?<=[@\w\d_])\s*([*]|/)', r_protected)
         if len(operands) > 1:
             operands = restore(operands)
-            if self.language is not Language.C:
-                operands = [operand.replace('@__f2py_DOUBLE_STAR@', '**')
-                            for operand in operands]
+            # Restore the ** operator
+            operands = [operand.replace('@__f2py_DOUBLE_STAR@', '**')
+                        for operand in operands]
             # Expression is an arithmetic product
```

Alternatively, the exponentiation handling block (lines 1432-1439) could be moved before the multiplication/division block, or the condition could be changed to check for `**` regardless of language mode.