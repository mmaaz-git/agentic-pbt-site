# Bug Report: pandas.io.formats.format._trim_zeros_complex Loses Closing Parenthesis

**Target**: `pandas.io.formats.format._trim_zeros_complex`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_trim_zeros_complex` function in `pandas.io.formats.format` incorrectly strips the closing parenthesis from complex number string representations that include parentheses (the standard Python format).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.formats.format import _trim_zeros_complex


@given(st.lists(st.tuples(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False)
), min_size=1, max_size=10))
def test_trim_zeros_complex_preserves_parentheses(float_pairs):
    values = [complex(r, i) for r, i in float_pairs]
    str_complexes = [str(v) for v in values]
    trimmed = _trim_zeros_complex(str_complexes)

    for original, result in zip(str_complexes, trimmed):
        if original.endswith(')'):
            assert result.endswith(')'), f"Lost closing parenthesis: {original} -> {result}"
```

**Failing input**: `[(1.0, 1.0)]` (creates complex `(1+1j)`)

## Reproducing the Bug

```python
from pandas.io.formats.format import _trim_zeros_complex

values = [complex(1, 2), complex(3, 4)]
str_values = [str(v) for v in values]

print(f"Input:  {str_values}")
result = _trim_zeros_complex(str_values)
print(f"Output: {result}")
```

**Output**:
```
Input:  ['(1+2j)', '(3+4j)']
Output: ['(1+2j', '(3+4j']
```

The closing `)` is lost from each complex number.

## Why This Is A Bug

Python's standard string representation of complex numbers includes parentheses when both real and imaginary parts are present: `(1+2j)`. The `_trim_zeros_complex` function is supposed to trim trailing zeros from the numeric parts while preserving the structure of the string representation. However, it incorrectly discards the closing parenthesis.

The issue is in the parsing logic:
1. The function splits the input using `re.split(r"([j+-])", x)`
2. For input `"(1+2j)"`, this produces: `['(1', '+', '2', 'j', ')']`
3. It takes `trimmed[:-4]` for the real part (gets `'(1'`)
4. It takes `trimmed[-4:-2]` for the imaginary part (gets `'+2'`)
5. The `)` at position `-1` is never included in the reconstruction

This affects DataFrame display when showing complex numbers with `fixed_width` formatting.

## Fix

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1767,8 +1767,15 @@ def _trim_zeros_complex(str_complexes: ArrayLike, decimal: str = ".") -> list[s
     """
     real_part, imag_part = [], []
+    has_parens = []
     for x in str_complexes:
+        # Track and remove parentheses for processing
+        paren = x.startswith('(') and x.endswith(')')
+        has_parens.append(paren)
+        if paren:
+            x = x[1:-1]
+
         # Complex numbers are represented as "(-)xxx(+/-)xxxj"
         # The split will give [{"", "-"}, "xxx", "+/-", "xxx", "j", ""]
         # Therefore, the imaginary part is the 4th and 3rd last elements,
@@ -1787,7 +1794,11 @@ def _trim_zeros_complex(str_complexes: ArrayLike, decimal: str = ".") -> list[s
         real_pt  # real part, possibly NaN
         + imag_pt[0]  # +/-
         + f"{imag_pt[1:]:>{padded_length}}"  # complex part (no sign), possibly nan
         + "j"
-        for real_pt, imag_pt in zip(padded_parts[:n], padded_parts[n:])
+        for i, (real_pt, imag_pt) in enumerate(zip(padded_parts[:n], padded_parts[n:]))
     ]
+
+    # Add back parentheses where they were present
+    padded = [f"({p})" if has_parens[i] else p for i, p in enumerate(padded)]
+
     return padded
```