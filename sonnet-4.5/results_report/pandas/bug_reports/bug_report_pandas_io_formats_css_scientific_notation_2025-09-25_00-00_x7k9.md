# Bug Report: pandas.io.formats.css.CSSResolver.size_to_pt Scientific Notation Parsing

**Target**: `pandas.io.formats.css.CSSResolver.size_to_pt`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CSSResolver.size_to_pt()` method cannot correctly parse CSS size values written in scientific notation (e.g., `1e-5pt`), causing incorrect conversions and precision loss.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.formats.css import CSSResolver
import math


@settings(max_examples=1000)
@given(st.floats(min_value=0.000001, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_pt_to_pt_should_preserve_value(value):
    resolver = CSSResolver()
    input_str = f"{value}pt"
    result = resolver.size_to_pt(input_str)
    result_val = float(result.rstrip("pt"))
    assert math.isclose(result_val, value, abs_tol=1e-5) or result_val == value
```

**Failing input**: `6.103515625e-05` (and any value that Python represents in scientific notation)

## Reproducing the Bug

```python
from pandas.io.formats.css import CSSResolver

resolver = CSSResolver()

print(resolver.size_to_pt("1e-5pt"))
print(resolver.size_to_pt("0.00001pt"))
```

**Expected output:**
```
0.00001pt
0.00001pt
```

**Actual output:**
```
0pt
0.00001pt
```

## Why This Is A Bug

The regex pattern `r"^(\S*?)([a-zA-Z%!].*)"` at line 351 of `css.py` is designed to split a CSS size value into a numeric part and a unit part. However, it incorrectly handles scientific notation:

- Input: `"1e-5pt"`
- Regex groups: `('1', 'e-5pt')` ❌
- Expected groups: `('1e-5', 'pt')` ✓

The letter 'e' is captured as the start of the unit instead of being part of the number. This causes:
1. The value `'1'` to be parsed instead of `'1e-5'`
2. The unit `'e-5pt'` to fail conversion lookups
3. The error handler to return a default value of `0pt`

This affects any CSS size value that Python formats in scientific notation, particularly:
- Very small values (< 0.0001)
- Very large values (> 1,000,000)
- Programmatically generated CSS where float-to-string conversion uses scientific notation

## Fix

```diff
--- a/pandas/io/formats/css.py
+++ b/pandas/io/formats/css.py
@@ -348,7 +348,7 @@ class CSSResolver:
             )
             return self.size_to_pt("1!!default", conversions=conversions)

-        match = re.match(r"^(\S*?)([a-zA-Z%!].*)", in_val)
+        match = re.match(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z%!].*)", in_val)
         if match is None:
             return _error()
```

The improved regex pattern:
- `[+-]?` - Optional sign
- `(?:\d+\.?\d*|\.\d+)` - Matches decimal numbers (e.g., `123`, `1.5`, `.5`)
- `(?:[eE][+-]?\d+)?` - Optional scientific notation exponent (e.g., `e-5`, `E+10`)
- `([a-zA-Z%!].*)` - Unit part (unchanged)