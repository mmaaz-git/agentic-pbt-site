# Bug Report: pandas.io.formats.css Excessive Trailing Zeros in Float PT Values

**Target**: `pandas.io.formats.css.CSSResolver.size_to_pt`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `size_to_pt` method in `CSSResolver` outputs non-integer pt values with excessive trailing zeros (e.g., `1.500000pt` instead of `1.5pt`), resulting in unnecessarily verbose CSS output.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.formats.css import CSSResolver
import re


@given(st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_non_integer_pt_values_formatting(pt_val):
    resolver = CSSResolver()
    result = resolver(f"font-size: {pt_val}pt")

    if 'font-size' in result:
        val = result['font-size']
        match = re.match(r'^(\d+(?:\.\d+)?)pt$', val)
        assert match, f"Expected pt value, got {val}"

        trailing_zeros_match = re.search(r'\.(\d*?)(0+)pt$', val)
        if trailing_zeros_match and len(trailing_zeros_match.group(2)) > 1:
            assert False, f"PT value has excessive trailing zeros: {val}"
```

**Failing input**: `pt_val=1.5`

## Reproducing the Bug

```python
from pandas.io.formats.css import CSSResolver

resolver = CSSResolver()

result = resolver("font-size: 1.5pt")
print(result['font-size'])

result = resolver("margin: 10px")
print(result['margin-top'])
```

Output:
```
1.500000pt
7.500000pt
```

Expected:
```
1.5pt
7.5pt
```

## Why This Is A Bug

The CSS resolver is designed to produce clean, readable CSS output for styling pandas DataFrames. Outputting `1.500000pt` instead of `1.5pt` violates the reasonable expectation of clean, minimal output. While mathematically correct, the excessive trailing zeros make the output unnecessarily verbose and harder to read, especially when inspecting styled DataFrame output in HTML or other formats.

## Fix

```diff
--- a/pandas/io/formats/css.py
+++ b/pandas/io/formats/css.py
@@ -378,10 +378,14 @@ class CSSResolver:
             val *= mul

         val = round(val, 5)
         if int(val) == val:
             size_fmt = f"{int(val):d}pt"
         else:
-            size_fmt = f"{val:f}pt"
+            size_fmt = f"{val:g}pt"
         return size_fmt
```

The fix changes the float formatting from `{val:f}` (which uses 6 decimal places by default) to `{val:g}` (which uses the shortest representation, removing trailing zeros).