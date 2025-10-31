# Bug Report: pandas.io.formats.css CSSResolver.size_to_pt Scientific Notation

**Target**: `pandas.io.formats.css.CSSResolver.size_to_pt`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `size_to_pt` method fails to parse CSS sizes specified in scientific notation (e.g., `"1e-5pt"`, `"2.5e3px"`), incorrectly treating them as invalid and falling back to a default value of `"0pt"`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.formats.css import CSSResolver

resolver = CSSResolver()

@given(
    val=st.floats(min_value=1e-10, max_value=1e10, allow_nan=False, allow_infinity=False),
    unit=st.sampled_from(["pt", "px", "em", "rem", "in", "cm", "mm"])
)
def test_size_to_pt_scientific_notation(val, unit):
    input_str = f"{val}{unit}"
    result = resolver.size_to_pt(input_str)
    assert result.endswith("pt"), f"Result {result} should end with 'pt'"
    result_val = float(result.rstrip("pt"))
    assert result_val != 0 or val == 0, f"Non-zero input {input_str} should not produce 0pt"
```

**Failing input**: `val=1e-5, unit="pt"`

## Reproducing the Bug

```python
from pandas.io.formats.css import CSSResolver

resolver = CSSResolver()

result = resolver.size_to_pt("1e-5pt")
print(f"Input:  1e-5pt")
print(f"Output: {result}")
print(f"Expected value: 1e-05")
print(f"Actual value:   {float(result.rstrip('pt'))}")

result = resolver.size_to_pt("2.5e3px")
print(f"\nInput:  2.5e3px")
print(f"Output: {result}")
```

Output:
```
Input:  1e-5pt
Output: 0pt
Expected value: 1e-05
Actual value:   0.0

Input:  2.5e3px
Output: 0pt
```

## Why This Is A Bug

The regex pattern at line 351 fails to correctly parse scientific notation:

```python
match = re.match(r"^(\S*?)([a-zA-Z%!].*)", in_val)
```

For input `"1e-5pt"`:
- The non-greedy `(\S*?)` matches just `"1"` (stopping at the first letter)
- The `([a-zA-Z%!].*)` matches `"e-5pt"` (starting with letter 'e')
- Result: `val="1"`, `unit="e-5pt"`
- The unit `"e-5pt"` is not in the conversions table, triggering the error handler

While scientific notation is less common in CSS, it's valid Python syntax and should be supported for programmatic generation of CSS values, especially for very small or large values.

## Fix

```diff
--- a/pandas/io/formats/css.py
+++ b/pandas/io/formats/css.py
@@ -348,7 +348,9 @@ class CSSResolver:
             )
             return self.size_to_pt("1!!default", conversions=conversions)

-        match = re.match(r"^(\S*?)([a-zA-Z%!].*)", in_val)
+        # Match numeric value (including scientific notation) followed by unit
+        # Use lookahead to ensure we capture full scientific notation (e.g., 1e-5)
+        match = re.match(r"^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)([a-zA-Z%!].*)", in_val)
         if match is None:
             return _error()