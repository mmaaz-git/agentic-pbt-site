# Bug Report: numpy.f2py.symbolic Power Operator Produces Invalid Syntax

**Target**: `numpy.f2py.symbolic.Expr.fromstring()` and `tostring()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `fromstring()` method with default C language mode incorrectly parses the power operator `**`, resulting in syntactically invalid output `x * *N` instead of valid power notation when converted back to string.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for numpy.f2py.symbolic power operator bug"""

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

if __name__ == "__main__":
    test_power_operator_roundtrip()
```

<details>

<summary>
**Failing input**: `exp=1, var='x'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 23, in <module>
    test_power_operator_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 10, in test_power_operator_roundtrip
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 16, in test_power_operator_roundtrip
    assert '**' in s or 'pow' in s.lower() or '^' in s, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Power operator lost in tostring: x**1 -> x * *1
Falsifying example: test_power_operator_roundtrip(
    # The test always failed when commented parts were varied together.
    exp=1,  # or any other generated value
    var='x',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Demonstrate the numpy.f2py.symbolic power operator bug"""

import numpy.f2py.symbolic as sym
from numpy.f2py.symbolic import Language

# Test case from bug report
e = sym.fromstring('x**2')
print(f'Input: x**2')
print(f'Parsed expression: {e!r}')
print(f'Fortran output: {e.tostring(language=Language.Fortran)}')
print(f'Python output: {e.tostring(language=Language.Python)}')
print(f'C output: {e.tostring(language=Language.C)}')
print()

# Test with explicit language specification
print('With explicit Fortran language:')
e_fortran = sym.fromstring('x**2', language=Language.Fortran)
print(f'Input: x**2 (parsed as Fortran)')
print(f'Parsed expression: {e_fortran!r}')
print(f'Fortran output: {e_fortran.tostring(language=Language.Fortran)}')
print()

# Test round-trip
print('Round-trip test:')
malformed = e.tostring(language=Language.Fortran)
print(f'Malformed string: {malformed!r}')
e2 = sym.fromstring(malformed)
print(f'Re-parsed: {e2!r}')
print(f'Re-parsed tostring: {e2.tostring(language=Language.Fortran)}')
print(f'Round-trip equality: {e == e2}')
```

<details>

<summary>
Output demonstrates syntactically invalid `x * *2` instead of `x**2`
</summary>
```
Input: x**2
Parsed expression: Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'x'): 1, Expr(Op.DEREF, Expr(Op.INTEGER, (2, 4))): 1})
Fortran output: x * *2
Python output: x * *2
C output: x * *2

With explicit Fortran language:
Input: x**2 (parsed as Fortran)
Parsed expression: Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'x'): 2})
Fortran output: x ** 2

Round-trip test:
Malformed string: 'x * *2'
Re-parsed: Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'x'): 1, Expr(Op.DEREF, Expr(Op.INTEGER, (2, 4))): 1})
Re-parsed tostring: x * *2
Round-trip equality: True
```
</details>

## Why This Is A Bug

The numpy.f2py.symbolic module is designed to parse and generate code for Fortran/C symbolic expressions. According to the module documentation (line 20 of symbolic.py), it explicitly supports "arithmetic expressions involving integers and operations like addition (+), subtraction (-), multiplication (*), division (Fortran / is Python //, Fortran // is concatenate), and **exponentiation (**).**"

The bug violates this specification in multiple ways:

1. **Invalid Syntax Generation**: The output `x * *2` is syntactically invalid in all three supported languages (C, Fortran, Python). It cannot be compiled or executed.

2. **Default Behavior Broken**: Using `fromstring('x**2')` with default parameters produces unusable code. Users expect reasonable defaults to work.

3. **Incorrect Internal Representation**: The expression is incorrectly parsed as a multiplication (`*`) followed by a dereference operator (`*2`), creating `Expr(Op.DEREF, Expr(Op.INTEGER, (2, 4)))` instead of proper exponentiation.

4. **Purpose Violation**: The module's purpose is code generation for f2py. Generating syntactically invalid code defeats this purpose entirely.

## Relevant Context

The root cause is in the `_FromStringWorker.process()` method at line 1433 of `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/f2py/symbolic.py`:

```python
# exponentiation operations
if self.language is not Language.C and '**' in r:
```

This condition prevents the parser from recognizing `**` as an exponentiation operator when parsing in C mode (the default). Since C doesn't have a `**` operator, the parser treats the two asterisks separately:
- First `*` as multiplication
- Second `*` as a dereference operator

The `tostring()` method then faithfully outputs this malformed representation as `x * *2`.

Documentation links:
- Module source: numpy/f2py/symbolic.py
- f2py documentation: https://numpy.org/doc/stable/f2py/

## Proposed Fix

The parser should either:
1. Recognize `**` as exponentiation even in C mode and convert to appropriate syntax on output, or
2. Reject `**` in C mode with a clear error message

Here's a proposed fix that implements option 1:

```diff
--- a/numpy/f2py/symbolic.py
+++ b/numpy/f2py/symbolic.py
@@ -1430,7 +1430,7 @@ class _FromStringWorker:
             return Expr(op, operand)

         # exponentiation operations
-        if self.language is not Language.C and '**' in r:
+        if '**' in r:
             operands = list(reversed(restore(r.split('**'))))
             result = self.process(operands[0])
             for operand in operands[1:]:
```

This allows the parser to recognize `**` as exponentiation regardless of language mode. The `tostring()` method already handles outputting the correct syntax for each language (lines 333-349): `pow(x, 2)` for C and `x ** 2` for Fortran/Python.