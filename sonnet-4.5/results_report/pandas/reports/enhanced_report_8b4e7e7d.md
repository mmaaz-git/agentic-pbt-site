# Bug Report: pandas.io.formats.css Excessive Trailing Zeros in Float PT Values

**Target**: `pandas.io.formats.css.CSSResolver.size_to_pt`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `size_to_pt` method in `CSSResolver` outputs non-integer pt values with exactly 6 decimal places, adding unnecessary trailing zeros (e.g., `1.500000pt` instead of `1.5pt`), resulting in unnecessarily verbose CSS output.

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


if __name__ == "__main__":
    test_non_integer_pt_values_formatting()
```

<details>

<summary>
**Failing input**: `pt_val=0.5`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 23, in <module>
    test_non_integer_pt_values_formatting()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 7, in test_non_integer_pt_values_formatting
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 19, in test_non_integer_pt_values_formatting
    assert False, f"PT value has excessive trailing zeros: {val}"
           ^^^^^
AssertionError: PT value has excessive trailing zeros: 0.500000pt
Falsifying example: test_non_integer_pt_values_formatting(
    pt_val=0.5,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/42/hypo.py:19
```
</details>

## Reproducing the Bug

```python
from pandas.io.formats.css import CSSResolver

resolver = CSSResolver()

# Test with direct pt values
result = resolver("font-size: 1.5pt")
print(f"Input: 'font-size: 1.5pt' -> Output: {result.get('font-size', 'None')}")

# Test with px values that convert to pt
result = resolver("margin: 10px")
print(f"Input: 'margin: 10px' -> Output (margin-top): {result.get('margin-top', 'None')}")

# Test with different decimal values
result = resolver("font-size: 3.75pt")
print(f"Input: 'font-size: 3.75pt' -> Output: {result.get('font-size', 'None')}")

# Test with integer pt value (should not have trailing zeros)
result = resolver("font-size: 5pt")
print(f"Input: 'font-size: 5pt' -> Output: {result.get('font-size', 'None')}")

# Test with small decimal value
result = resolver("font-size: 0.5pt")
print(f"Input: 'font-size: 0.5pt' -> Output: {result.get('font-size', 'None')}")

# Test with value that rounds
result = resolver("font-size: 1.3333333pt")
print(f"Input: 'font-size: 1.3333333pt' -> Output: {result.get('font-size', 'None')}")
```

<details>

<summary>
Output shows excessive trailing zeros for all non-integer values
</summary>
```
Input: 'font-size: 1.5pt' -> Output: 1.500000pt
Input: 'margin: 10px' -> Output (margin-top): 7.500000pt
Input: 'font-size: 3.75pt' -> Output: 3.750000pt
Input: 'font-size: 5pt' -> Output: 5pt
Input: 'font-size: 0.5pt' -> Output: 0.500000pt
Input: 'font-size: 1.3333333pt' -> Output: 1.333330pt
```
</details>

## Why This Is A Bug

The CSS resolver violates the principle of minimal, clean output that is already established in the code itself. The method `size_to_pt` contains special handling at lines 381-382 to format integer values without decimal points (e.g., `5pt` instead of `5.000000pt`), demonstrating that clean formatting is an intentional design goal. However, line 384 uses Python's `f` format specifier which defaults to 6 decimal places for all non-integer values, creating an inconsistency where integers get clean formatting but decimals always get exactly 6 decimal places regardless of necessity.

This contradicts CSS conventions where minimal representation is preferred (e.g., `1.5pt` is standard, not `1.500000pt`). While both formats are functionally equivalent in CSS, the excessive precision serves no purpose since the values are already rounded to 5 decimal places at line 380, yet displayed with 6. This makes the styled DataFrame output unnecessarily verbose and harder to read when inspecting HTML or debugging styles.

## Relevant Context

The `CSSResolver` class is an internal utility in `pandas.io.formats.css` used for parsing and resolving CSS styles, primarily when converting styled DataFrames to formats like Excel or HTML. The `size_to_pt` method converts various CSS units (px, em, rem, etc.) to point (pt) values.

Key code location: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/formats/css.py`, lines 380-385

The issue occurs because:
1. Line 380: Values are rounded to 5 decimal places
2. Line 381-382: Integer values get special formatting without decimals
3. Line 384: Non-integers use `f"{val:f}pt"` which defaults to 6 decimal places

Documentation: This is an internal API not covered in the public pandas documentation. The class is primarily used through the DataFrame styling interface documented at https://pandas.pydata.org/docs/user_guide/style.html

## Proposed Fix

```diff
--- a/pandas/io/formats/css.py
+++ b/pandas/io/formats/css.py
@@ -381,7 +381,7 @@ class CSSResolver:
         if int(val) == val:
             size_fmt = f"{int(val):d}pt"
         else:
-            size_fmt = f"{val:f}pt"
+            size_fmt = f"{val:g}pt"
         return size_fmt
```