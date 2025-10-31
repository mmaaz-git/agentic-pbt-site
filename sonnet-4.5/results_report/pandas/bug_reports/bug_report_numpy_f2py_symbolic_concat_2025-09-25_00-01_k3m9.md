# Bug Report: numpy.f2py.symbolic String Concatenation Not Associative

**Target**: `numpy.f2py.symbolic.normalize`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalize` function fails to flatten nested `CONCAT` expressions, violating the mathematical property that string concatenation is associative: `(a // b) // c == a // (b // c)`.

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
def test_string_concat_associative(a, b, c):
    result1 = (a // b) // c
    result2 = a // (b // c)
    assert normalize(result1) == normalize(result2)
```

**Failing input**: `a=Expr(Op.STRING, ('""', 1)), b=Expr(Op.STRING, ('""', 1)), c=Expr(Op.STRING, ("''", 1))`

## Reproducing the Bug

```python
from numpy.f2py.symbolic import as_string, normalize

a = as_string('""', 1)
b = as_string('""', 1)
c = as_string("''", 1)

result1 = (a // b) // c
result2 = a // (b // c)

print(f"(a // b) // c = {repr(normalize(result1))}")
print(f"a // (b // c) = {repr(normalize(result2))}")
print(f"Equal: {normalize(result1) == normalize(result2)}")
```

**Output:**
```
(a // b) // c = Expr(Op.CONCAT, (Expr(Op.STRING, ('""', 1)), Expr(Op.STRING, ("''", 1))))
a // (b // c) = Expr(Op.CONCAT, (Expr(Op.STRING, ('""', 1)), Expr(Op.CONCAT, (Expr(Op.STRING, ('""', 1)), Expr(Op.STRING, ("''", 1))))))
Equal: False
```

## Why This Is A Bug

String concatenation is mathematically associative - the order of operations shouldn't matter for the final result. The `normalize` function successfully merges adjacent strings with the same quote type (line 899-916 in symbolic.py), but it fails to flatten nested `CONCAT` operations.

In this example:
- `(a // b) // c`: First concatenates two `""` strings (which merge to one `""`), then concatenates with `''`, resulting in a flat structure
- `a // (b // c)`: First creates a nested CONCAT for `(b // c)`, then wraps it in another CONCAT, resulting in nested structure

The normalized forms should be identical, but they're not because nested CONCAT nodes aren't flattened.

## Fix

The `normalize` function for `Op.CONCAT` should flatten nested CONCAT expressions before processing:

```diff
--- a/numpy/f2py/symbolic.py
+++ b/numpy/f2py/symbolic.py
@@ -898,7 +898,15 @@ def normalize(obj):

     if obj.op is Op.CONCAT:
         lst = [obj.data[0]]
-        for s in obj.data[1:]:
+        # Flatten nested CONCAT expressions first
+        flattened_data = []
+        for item in obj.data:
+            if item.op is Op.CONCAT:
+                flattened_data.extend(item.data)
+            else:
+                flattened_data.append(item)
+        lst = [flattened_data[0]]
+        for s in flattened_data[1:]:
             last = lst[-1]
             if (
                     last.op is Op.STRING
```