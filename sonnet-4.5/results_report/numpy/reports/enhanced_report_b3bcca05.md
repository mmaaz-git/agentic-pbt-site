# Bug Report: numpy.f2py.symbolic String Concatenation Violates Associativity

**Target**: `numpy.f2py.symbolic.normalize`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalize` function in numpy.f2py.symbolic fails to properly flatten nested CONCAT expressions, causing mathematically equivalent string concatenation expressions to produce different normalized forms, violating the associative property of concatenation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from numpy.f2py.symbolic import as_string, normalize


@st.composite
def expr_strings(draw):
    s = draw(st.text(min_size=0, max_size=20))
    quote_char = draw(st.sampled_from(['"', "'"]))
    quoted = quote_char + s + quote_char
    return as_string(quoted, kind=1)


@given(expr_strings(), expr_strings(), expr_strings())
@settings(max_examples=500)
def test_concat_associativity(a, b, c):
    """Test that string concatenation is associative: (a // b) // c == a // (b // c)"""
    result1 = (a // b) // c
    result2 = a // (b // c)
    assert normalize(result1) == normalize(result2), \
        f"Associativity violated: normalize({repr(result1)}) != normalize({repr(result2)})"


if __name__ == "__main__":
    # Run the test
    test_concat_associativity()
```

<details>

<summary>
**Failing input**: `a=Expr(Op.STRING, ('""', 1)), b=Expr(Op.STRING, ('""', 1)), c=Expr(Op.STRING, ("''", 1))`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 25, in <module>
    test_concat_associativity()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 14, in test_concat_associativity
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 19, in test_concat_associativity
    assert normalize(result1) == normalize(result2), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Associativity violated: normalize(Expr(Op.CONCAT, (Expr(Op.STRING, ('""', 1)), Expr(Op.STRING, ("''", 1))))) != normalize(Expr(Op.CONCAT, (Expr(Op.STRING, ('""', 1)), Expr(Op.CONCAT, (Expr(Op.STRING, ('""', 1)), Expr(Op.STRING, ("''", 1)))))))
Falsifying example: test_concat_associativity(
    a=Expr(Op.STRING, ('""', 1)),  # or any other generated value
    b=Expr(Op.STRING, ('""', 1)),  # or any other generated value
    c=Expr(Op.STRING, ("''", 1)),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/21/hypo.py:20
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:913
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py:916
```
</details>

## Reproducing the Bug

```python
from numpy.f2py.symbolic import as_string, normalize

# Create three string expressions
a = as_string('""', 1)  # Empty string with double quotes
b = as_string('""', 1)  # Empty string with double quotes
c = as_string("''", 1)  # Empty string with single quotes

# Test associativity: (a // b) // c vs a // (b // c)
result1 = (a // b) // c
result2 = a // (b // c)

# Normalize both results
normalized1 = normalize(result1)
normalized2 = normalize(result2)

print("Testing string concatenation associativity in numpy.f2py.symbolic")
print("=" * 70)
print()
print("Input strings:")
print(f"  a = {repr(a)}")
print(f"  b = {repr(b)}")
print(f"  c = {repr(c)}")
print()
print("Expression 1: (a // b) // c")
print(f"  Raw: {repr(result1)}")
print(f"  Normalized: {repr(normalized1)}")
print()
print("Expression 2: a // (b // c)")
print(f"  Raw: {repr(result2)}")
print(f"  Normalized: {repr(normalized2)}")
print()
print("Normalized forms equal?", normalized1 == normalized2)
print()
print("Expected: Both normalized forms should be equal (associativity)")
print("Actual: They are different, violating associativity")
```

<details>

<summary>
AssertionError: Normalized forms are not equal
</summary>
```
Testing string concatenation associativity in numpy.f2py.symbolic
======================================================================

Input strings:
  a = Expr(Op.STRING, ('""', 1))
  b = Expr(Op.STRING, ('""', 1))
  c = Expr(Op.STRING, ("''", 1))

Expression 1: (a // b) // c
  Raw: Expr(Op.CONCAT, (Expr(Op.STRING, ('""', 1)), Expr(Op.STRING, ("''", 1))))
  Normalized: Expr(Op.CONCAT, (Expr(Op.STRING, ('""', 1)), Expr(Op.STRING, ("''", 1))))

Expression 2: a // (b // c)
  Raw: Expr(Op.CONCAT, (Expr(Op.STRING, ('""', 1)), Expr(Op.CONCAT, (Expr(Op.STRING, ('""', 1)), Expr(Op.STRING, ("''", 1))))))
  Normalized: Expr(Op.CONCAT, (Expr(Op.STRING, ('""', 1)), Expr(Op.CONCAT, (Expr(Op.STRING, ('""', 1)), Expr(Op.STRING, ("''", 1))))))

Normalized forms equal? False

Expected: Both normalized forms should be equal (associativity)
Actual: They are different, violating associativity
```
</details>

## Why This Is A Bug

String concatenation in Fortran (the `//` operator) is mathematically associative - `(a // b) // c` should produce the same result as `a // (b // c)`. The `normalize` function's purpose, as stated in its docstring, is to "Normalize Expr and apply basic evaluation methods." This means it should produce canonical forms for equivalent expressions.

The bug occurs because the `normalize` function (symbolic.py:899-916) successfully merges adjacent STRING expressions with matching quote types, but it fails to first flatten nested CONCAT operations. This leads to different normalized forms for mathematically equivalent expressions:

1. **Expression 1: `(a // b) // c`**: When `a` and `b` (both `""`) are concatenated first, they merge into a single `""` string. Then concatenating with `c` (`''`) produces a flat `CONCAT(STRING(""), STRING(''))`

2. **Expression 2: `a // (b // c)`**: When `b` and `c` are concatenated first (different quotes, no merge), we get `CONCAT(b, c)`. Then concatenating `a` with this nested structure produces `CONCAT(STRING(""), CONCAT(STRING(""), STRING('')))`

This violates the Fortran language specification where the `//` operator is associative, and it breaks the normalization contract where equivalent expressions should have identical normalized forms.

## Relevant Context

The numpy.f2py module is used to generate Python interfaces for Fortran code. The symbolic module parses and manipulates Fortran expressions, including string operations. Proper normalization is crucial for:

- Correct interface generation when Fortran code uses string concatenation
- Symbolic analysis and optimization of Fortran expressions
- Maintaining semantic equivalence during code transformation

The normalize function already handles other associative operations correctly (e.g., TERMS for addition, FACTORS for multiplication) by using dictionaries to accumulate terms/factors. The CONCAT operation needs similar flattening logic.

## Proposed Fix

```diff
--- a/numpy/f2py/symbolic.py
+++ b/numpy/f2py/symbolic.py
@@ -897,8 +897,17 @@ def normalize(obj):
         return as_apply(ArithOp.DIV, numer, denom)

     if obj.op is Op.CONCAT:
-        lst = [obj.data[0]]
-        for s in obj.data[1:]:
+        # First, flatten any nested CONCAT operations
+        flattened_data = []
+        for item in obj.data:
+            if isinstance(item, Expr) and item.op is Op.CONCAT:
+                flattened_data.extend(item.data)
+            else:
+                flattened_data.append(item)
+
+        # Then merge adjacent strings with matching quotes
+        lst = [flattened_data[0]]
+        for s in flattened_data[1:]:
             last = lst[-1]
             if (
                     last.op is Op.STRING
```