# Bug Report: pandas.io.formats.css.CSSResolver.size_to_pt Scientific Notation Parsing Failure

**Target**: `pandas.io.formats.css.CSSResolver.size_to_pt`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `size_to_pt` method in `CSSResolver` fails to parse CSS size values in scientific notation (e.g., `1e-10pt`, `1.5e-20px`), incorrectly treating them as "Unhandled size" and returning `0pt` instead of the proper conversion.

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

if __name__ == "__main__":
    test_css_size_to_pt_always_returns_pt()
```

<details>

<summary>
**Failing input**: Scientific notation values like `1.8163872669965612e-177pt`, `5.960464477539063e-08px`
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/23/hypo.py:18: CSSWarning: Unhandled size: '1.8163872669965612e-177pt'
  result = resolver.size_to_pt(input_str)
/home/npc/pbt/agentic-pbt/worker_/23/hypo.py:18: CSSWarning: Unhandled size: '1.8163872669965612e-177em'
  result = resolver.size_to_pt(input_str)
/home/npc/pbt/agentic-pbt/worker_/23/hypo.py:18: CSSWarning: Unhandled size: '8.582673628887381e-180rem'
  result = resolver.size_to_pt(input_str)
/home/npc/pbt/agentic-pbt/worker_/23/hypo.py:18: CSSWarning: Unhandled size: '2.225073858507203e-309px'
  result = resolver.size_to_pt(input_str)
[... 283 more warnings for various scientific notation values ...]
```
</details>

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
        if len(w) > 0:
            print(f"  Warning message: {w[0].message}")
```

<details>

<summary>
All inputs incorrectly return 0pt with "Unhandled size" warnings
</summary>
```
Input: 1e-10pt, Result: 0pt, Warnings: True
  Warning message: Unhandled size: '1e-10pt'
Input: 1.5e-20px, Result: 0pt, Warnings: True
  Warning message: Unhandled size: '1.5e-20px'
Input: 3e-100em, Result: 0pt, Warnings: True
  Warning message: Unhandled size: '3e-100em'
```
</details>

## Why This Is A Bug

This violates the CSS specification and expected behavior for CSS parsers. According to the W3C CSS Values and Units Module Level 4 specification, scientific notation is explicitly supported in CSS numeric values: "When written literally, a number is either an integer, or zero or more decimal digits followed by a dot (.) followed by one or more decimal digits; optionally, it can be concluded by the letter 'e' or 'E' followed by an integer indicating the base-ten exponent in scientific notation."

The bug occurs because the regex pattern `r"^(\S*?)([a-zA-Z%!].*)"` on line 351 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/formats/css.py` uses a non-greedy quantifier `\S*?` that stops at the first letter character. For scientific notation like `1e-10pt`:
- Current behavior: matches value=`1`, unit=`e-10pt`
- Expected behavior: should match value=`1e-10`, unit=`pt`

Since `e-10pt` is not a recognized CSS unit in the UNIT_RATIOS dictionary, the function calls `_error()`, issues a warning about "Unhandled size", and returns `0pt`. This causes silent data corruption where valid CSS values are converted to zero, potentially breaking styling without obvious failure.

## Relevant Context

The `size_to_pt` function is part of pandas' CSS utilities for "interpreting CSS from Stylers for formatting non-HTML outputs." It's designed to convert various CSS units (px, em, rem, etc.) to points (pt) for consistent rendering. The function uses a dictionary of unit conversion ratios:

```python
UNIT_RATIOS = {
    "pt": ("pt", 1),
    "em": ("em", 1),
    "rem": ("rem", 1),
    "%": ("%", 1),
    "ex": ("em", 0.5),
    "px": ("pt", 0.75),
    "pc": ("pt", 12),
    "in": ("pt", 72),
    "cm": ("pt", 72 / 2.54),
    "mm": ("pt", 72 / 25.4),
    "q": ("pt", 72 / 101.6),
    "!!default": ("em", 0),
}
```

MDN documentation confirms scientific notation is valid CSS: "A `<number>` can also end with the letter e or E followed by an integer, which indicates a base-ten exponent in scientific notation."

While scientific notation may be uncommon in hand-written CSS, it can appear in generated stylesheets, mathematical computations, or when dealing with very small/large values.

## Proposed Fix

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