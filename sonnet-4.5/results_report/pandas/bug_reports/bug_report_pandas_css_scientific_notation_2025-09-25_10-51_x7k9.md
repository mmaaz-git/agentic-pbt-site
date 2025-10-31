# Bug Report: pandas.io.formats.css.CSSResolver.size_to_pt Scientific Notation Parsing

**Target**: `pandas.io.formats.css.CSSResolver.size_to_pt`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `size_to_pt` method in `CSSResolver` incorrectly parses CSS size values in scientific notation (e.g., `1e-10pt`, `1.5e-20px`), treating them as "Unhandled size" and returning incorrect results (0pt instead of the proper conversion).

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, assume, settings
import pandas.io.formats.css as css


@given(
    value=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    unit=st.sampled_from(['pt', 'px', 'em', 'rem'])
)
@settings(max_examples=500)
def test_css_size_to_pt_always_returns_pt(value, unit):
    resolver = css.CSSResolver()
    if value < 0:
        assume(False)
    input_str = f"{value}{unit}"
    result = resolver.size_to_pt(input_str)
    assert result.endswith('pt'), f"Result should end with 'pt': {result}"
```

**Failing input**: `"1e-10pt"`, `"1.5e-20px"`, `"3e-100em"`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas.io.formats.css as css
import warnings

resolver = css.CSSResolver()

test_cases = ["1e-10pt", "1.5e-20px", "3e-100em"]

for test_case in test_cases:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = resolver.size_to_pt(test_case)
        print(f"Input: {test_case}, Result: {result}, Warnings: {len(w) > 0}")
```

Output:
```
Input: 1e-10pt, Result: 0pt, Warnings: True
Input: 1.5e-20px, Result: 0pt, Warnings: True
Input: 3e-100em, Result: 0pt, Warnings: True
```

All inputs are treated as "Unhandled size" and return `0pt` instead of the correct conversion.

## Why This Is A Bug

The regex pattern on line 351 of css.py is `r"^(\S*?)([a-zA-Z%!].*)"`. The non-greedy quantifier `\S*?` stops at the first character that could start the unit part (any letter). For scientific notation like `1e-10pt`:
- The pattern incorrectly matches: value=`1`, unit=`e-10pt`
- Should match: value=`1e-10`, unit=`pt`

Since `e-10pt` is not a valid unit, the function calls `_error()`, warns about "Unhandled size", and returns `0pt`.

Scientific notation is valid in CSS values, and this bug causes silent incorrect conversions for very small or very large values.

## Fix

```diff
--- a/pandas/io/formats/css.py
+++ b/pandas/io/formats/css.py
@@ -348,7 +348,7 @@ class CSSResolver:
             )
             return self.size_to_pt("1!!default", conversions=conversions)

-        match = re.match(r"^(\S*?)([a-zA-Z%!].*)", in_val)
+        match = re.match(r"^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)([a-zA-Z%!].*)", in_val)
         if match is None:
             return _error()
```

This change uses a proper regex pattern for floating point numbers including scientific notation:
- `[-+]?` - optional leading sign
- `[0-9]*\.?[0-9]+` - digits with optional decimal point
- `(?:[eE][-+]?[0-9]+)?` - optional exponent part (e or E followed by optional sign and digits)

This correctly parses scientific notation without incorrectly capturing the 'e' from unit names like 'em'.