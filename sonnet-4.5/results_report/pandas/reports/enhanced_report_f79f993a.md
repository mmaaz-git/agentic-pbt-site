# Bug Report: pandas.io.formats.css.CSSResolver.size_to_pt Scientific Notation Parsing Failure

**Target**: `pandas.io.formats.css.CSSResolver.size_to_pt`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CSSResolver.size_to_pt()` method fails to correctly parse CSS size values written in scientific notation, returning `"0pt"` instead of the correct converted value, violating CSS specification requirements.

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


if __name__ == "__main__":
    test_pt_to_pt_should_preserve_value()
```

<details>

<summary>
<b>Failing input</b>: <code>6.103515625e-05</code>
</summary>

```
/home/npc/pbt/agentic-pbt/worker_/57/hypo.py:11: CSSWarning: Unhandled size: '1e-06pt'
  result = resolver.size_to_pt(input_str)
/home/npc/pbt/agentic-pbt/worker_/57/hypo.py:11: CSSWarning: Unhandled size: '6.103515625e-05pt'
  result = resolver.size_to_pt(input_str)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 17, in <module>
    test_pt_to_pt_should_preserve_value()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 7, in test_pt_to_pt_should_preserve_value
    @given(st.floats(min_value=0.000001, max_value=1e6, allow_nan=False, allow_infinity=False))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 13, in test_pt_to_pt_should_preserve_value
    assert math.isclose(result_val, value, abs_tol=1e-5) or result_val == value
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_pt_to_pt_should_preserve_value(
    value=6.103515625e-05,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/css.py:366
```
</details>

## Reproducing the Bug

```python
from pandas.io.formats.css import CSSResolver

# Create a resolver instance
resolver = CSSResolver()

# Test cases with scientific notation
print("Testing scientific notation CSS size parsing:")
print("=" * 50)

# Test case 1: Small value in scientific notation
input1 = "1e-5pt"
result1 = resolver.size_to_pt(input1)
print(f"Input:  '{input1}'")
print(f"Result: '{result1}'")
print(f"Expected: '0.00001pt'")
print()

# Test case 2: Same value in decimal notation
input2 = "0.00001pt"
result2 = resolver.size_to_pt(input2)
print(f"Input:  '{input2}'")
print(f"Result: '{result2}'")
print(f"Expected: '0.00001pt'")
print()

# Test case 3: Specific failing value from property test
input3 = "6.103515625e-05pt"
result3 = resolver.size_to_pt(input3)
print(f"Input:  '{input3}'")
print(f"Result: '{result3}'")
print(f"Expected: '0.00006103515625pt' or '0.00006pt'")
print()

# Test case 4: Large value in scientific notation
input4 = "1e6pt"
result4 = resolver.size_to_pt(input4)
print(f"Input:  '{input4}'")
print(f"Result: '{result4}'")
print(f"Expected: '1000000pt'")
print()

# Test case 5: Scientific notation with decimal
input5 = "1.5e-3pt"
result5 = resolver.size_to_pt(input5)
print(f"Input:  '{input5}'")
print(f"Result: '{result5}'")
print(f"Expected: '0.0015pt'")
```

<details>

<summary>
CSSWarning: Unhandled size - All scientific notation inputs fail and return '0pt'
</summary>

```
/home/npc/pbt/agentic-pbt/worker_/57/repo.py:12: CSSWarning: Unhandled size: '1e-5pt'
  result1 = resolver.size_to_pt(input1)
/home/npc/pbt/agentic-pbt/worker_/57/repo.py:28: CSSWarning: Unhandled size: '6.103515625e-05pt'
  result3 = resolver.size_to_pt(input3)
/home/npc/pbt/agentic-pbt/worker_/57/repo.py:36: CSSWarning: Unhandled size: '1e6pt'
  result4 = resolver.size_to_pt(input4)
/home/npc/pbt/agentic-pbt/worker_/57/repo.py:44: CSSWarning: Unhandled size: '1.5e-3pt'
  result5 = resolver.size_to_pt(input5)
Testing scientific notation CSS size parsing:
==================================================
Input:  '1e-5pt'
Result: '0pt'
Expected: '0.00001pt'

Input:  '0.00001pt'
Result: '0.000010pt'
Expected: '0.00001pt'

Input:  '6.103515625e-05pt'
Result: '0pt'
Expected: '0.00006103515625pt' or '0.00006pt'

Input:  '1e6pt'
Result: '0pt'
Expected: '1000000pt'

Input:  '1.5e-3pt'
Result: '0pt'
Expected: '0.0015pt'
```
</details>

## Why This Is A Bug

This violates the CSS specification which explicitly requires support for scientific notation in numeric values. The W3C CSS Values and Units Module Level 3 states that numbers can use "the letter 'e' or 'E' followed by an integer indicating the base-ten exponent."

The bug occurs at line 351 in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/formats/css.py`:

```python
match = re.match(r"^(\S*?)([a-zA-Z%!].*)", in_val)
```

This regex pattern incorrectly parses scientific notation by treating the 'e' or 'E' in the exponent as the beginning of the unit string:

- Input: `"1e-5pt"`
- Current regex captures: `val='1'`, `unit='e-5pt'` ❌
- Should capture: `val='1e-5'`, `unit='pt'` ✓

This causes:
1. Only the mantissa (`'1'`) is parsed as the numeric value, ignoring the exponent
2. The unit is misidentified as `'e-5pt'` which doesn't exist in the conversion table
3. The error handler returns a default value of `'0pt'` with a warning

The function silently corrupts data by returning `'0pt'` instead of properly parsing valid CSS values, which could lead to incorrect styling in pandas-generated HTML output when values are programmatically generated in scientific notation.

## Relevant Context

The CSS specification (W3C CSS Syntax Module Level 3) defines the numeric token pattern as:
```
[+-]?(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?
```

This is a core CSS feature, not an edge case. Python automatically formats certain float values in scientific notation (e.g., values < 0.0001 or > 1e6), so this bug will be encountered when CSS values are programmatically generated from Python floats.

Relevant documentation:
- W3C CSS Values: https://www.w3.org/TR/css-values-3/#numeric-types
- W3C CSS Syntax: https://www.w3.org/TR/css-syntax-3/#consume-number
- Pandas CSS code: https://github.com/pandas-dev/pandas/blob/main/pandas/io/formats/css.py#L351

## Proposed Fix

Replace the incorrect regex pattern with one that properly handles scientific notation:

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

The improved regex pattern correctly handles:
- Optional sign: `[+-]?`
- Decimal numbers: `(?:\d+\.?\d*|\.\d+)` (matches `123`, `1.5`, `.5`)
- Optional scientific notation: `(?:[eE][+-]?\d+)?` (matches `e-5`, `E+10`)
- Unit part: `([a-zA-Z%!].*)` (unchanged from original)