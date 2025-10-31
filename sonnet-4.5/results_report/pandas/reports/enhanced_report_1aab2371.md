# Bug Report: pandas.io.formats.css CSSResolver.size_to_pt Scientific Notation Parsing Failure

**Target**: `pandas.io.formats.css.CSSResolver.size_to_pt`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `size_to_pt` method incorrectly parses CSS size values expressed in scientific notation (e.g., `"1e-5pt"`, `"2.5e3px"`), treating them as invalid and returning `"0pt"` instead of the correct converted value.

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

# Run the test
if __name__ == "__main__":
    test_size_to_pt_scientific_notation()
```

<details>

<summary>
**Failing input**: `val=1e-10, unit='pt'`
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/19/hypo.py:12: CSSWarning: Unhandled size: '1e-10pt'
  result = resolver.size_to_pt(input_str)
/home/npc/pbt/agentic-pbt/worker_/19/hypo.py:12: CSSWarning: Unhandled size: '1e-10em'
  result = resolver.size_to_pt(input_str)
/home/npc/pbt/agentic-pbt/worker_/19/hypo.py:12: CSSWarning: Unhandled size: '1e-10rem'
  result = resolver.size_to_pt(input_str)
/home/npc/pbt/agentic-pbt/worker_/19/hypo.py:12: CSSWarning: Unhandled size: '1e-10px'
  result = resolver.size_to_pt(input_str)
/home/npc/pbt/agentic-pbt/worker_/19/hypo.py:12: CSSWarning: Unhandled size: '1e-10mm'
  result = resolver.size_to_pt(input_str)
/home/npc/pbt/agentic-pbt/worker_/19/hypo.py:12: CSSWarning: Unhandled size: '1e-10in'
  result = resolver.size_to_pt(input_str)
/home/npc/pbt/agentic-pbt/worker_/19/hypo.py:12: CSSWarning: Unhandled size: '1e-10cm'
  result = resolver.size_to_pt(input_str)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 19, in <module>
    test_size_to_pt_scientific_notation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 7, in test_size_to_pt_scientific_notation
    val=st.floats(min_value=1e-10, max_value=1e10, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 15, in test_size_to_pt_scientific_notation
    assert result_val != 0 or val == 0, f"Non-zero input {input_str} should not produce 0pt"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Non-zero input 1e-10pt should not produce 0pt
Falsifying example: test_size_to_pt_scientific_notation(
    val=1e-10,
    unit='pt',  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/css.py:376
```
</details>

## Reproducing the Bug

```python
from pandas.io.formats.css import CSSResolver

resolver = CSSResolver()

# Test scientific notation inputs that should work but fail
print("=" * 60)
print("Testing scientific notation CSS size parsing")
print("=" * 60)

# Test case 1: Scientific notation with 'pt' unit
result = resolver.size_to_pt("1e-5pt")
print(f"\nTest 1: Scientific notation 1e-5pt")
print(f"  Input:    '1e-5pt'")
print(f"  Output:   '{result}'")
print(f"  Expected: '0.00001pt' (or equivalent)")
print(f"  Actual value: {float(result.rstrip('pt'))}")

# Test case 2: Scientific notation with 'px' unit (should convert to pt)
result = resolver.size_to_pt("2.5e3px")
print(f"\nTest 2: Scientific notation 2.5e3px")
print(f"  Input:    '2.5e3px'")
print(f"  Output:   '{result}'")
print(f"  Expected: '1875pt' (2500 * 0.75 conversion)")
actual_val = float(result.rstrip('pt'))
print(f"  Actual value: {actual_val}")

# Test case 3: Compare with non-scientific notation equivalents
print("\n" + "=" * 60)
print("Comparison with equivalent non-scientific notation")
print("=" * 60)

# Test the same values without scientific notation
result_regular = resolver.size_to_pt("0.00001pt")
print(f"\nRegular notation equivalent of 1e-5pt:")
print(f"  Input:    '0.00001pt'")
print(f"  Output:   '{result_regular}'")
print(f"  Value:    {float(result_regular.rstrip('pt'))}")

result_regular = resolver.size_to_pt("2500px")
print(f"\nRegular notation equivalent of 2.5e3px:")
print(f"  Input:    '2500px'")
print(f"  Output:   '{result_regular}'")
print(f"  Value:    {float(result_regular.rstrip('pt'))}")

# Test case 4: Additional scientific notation formats
print("\n" + "=" * 60)
print("Additional scientific notation tests")
print("=" * 60)

test_cases = [
    "1.23e-2em",
    "5E+2pt",
    "3.14E-1px",
]

for test in test_cases:
    result = resolver.size_to_pt(test)
    print(f"\nInput:  '{test}'")
    print(f"Output: '{result}'")
    try:
        print(f"Value:  {float(result.rstrip('pt'))}")
    except:
        print(f"Value:  <unable to parse>")
```

<details>

<summary>
CSSWarning: Unhandled size errors for all scientific notation inputs
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/19/repo.py:11: CSSWarning: Unhandled size: '1e-5pt'
  result = resolver.size_to_pt("1e-5pt")
/home/npc/pbt/agentic-pbt/worker_/19/repo.py:19: CSSWarning: Unhandled size: '2.5e3px'
  result = resolver.size_to_pt("2.5e3px")
/home/npc/pbt/agentic-pbt/worker_/19/repo.py:57: CSSWarning: Unhandled size: '1.23e-2em'
  result = resolver.size_to_pt(test)
/home/npc/pbt/agentic-pbt/worker_/19/repo.py:57: CSSWarning: Unhandled size: '5E+2pt'
  result = resolver.size_to_pt(test)
/home/npc/pbt/agentic-pbt/worker_/19/repo.py:57: CSSWarning: Unhandled size: '3.14E-1px'
  result = resolver.size_to_pt(test)
============================================================
Testing scientific notation CSS size parsing
============================================================

Test 1: Scientific notation 1e-5pt
  Input:    '1e-5pt'
  Output:   '0pt'
  Expected: '0.00001pt' (or equivalent)
  Actual value: 0.0

Test 2: Scientific notation 2.5e3px
  Input:    '2.5e3px'
  Output:   '0pt'
  Expected: '1875pt' (2500 * 0.75 conversion)
  Actual value: 0.0

============================================================
Comparison with equivalent non-scientific notation
============================================================

Regular notation equivalent of 1e-5pt:
  Input:    '0.00001pt'
  Output:   '0.000010pt'
  Value:    1e-05

Regular notation equivalent of 2.5e3px:
  Input:    '2500px'
  Output:   '1875pt'
  Value:    1875.0

============================================================
Additional scientific notation tests
============================================================

Input:  '1.23e-2em'
Output: '0pt'
Value:  0.0

Input:  '5E+2pt'
Output: '0pt'
Value:  0.0

Input:  '3.14E-1px'
Output: '0pt'
Value:  0.0
```
</details>

## Why This Is A Bug

The W3C CSS specification explicitly allows scientific notation for numeric values. According to the CSS Syntax Module Level 3, numbers can be "concluded by the letter 'e' or 'E' followed by an integer indicating the base-ten exponent." The current implementation fails to parse this valid CSS syntax due to a flawed regex pattern at line 351 of `pandas/io/formats/css.py`.

The regex `r"^(\S*?)([a-zA-Z%!].*)"` incorrectly splits scientific notation values:
- For input `"1e-5pt"`: The pattern matches `val="1"` and `unit="e-5pt"`
- For input `"2.5e3px"`: The pattern matches `val="2.5"` and `unit="e3px"`

The 'e' or 'E' in scientific notation is incorrectly interpreted as the beginning of the unit string, rather than part of the numeric value. Since units like `"e-5pt"` or `"e3px"` don't exist in the conversions table, the method falls back to returning `"0pt"` with a warning.

## Relevant Context

The `CSSResolver` class is an internal utility used by pandas for CSS parsing in its table styling features (accessible via the public `Styler` API). While direct usage of this class is not documented, it processes CSS that could come from user input through the Styler interface.

Scientific notation, while uncommon in hand-written CSS, is valid CSS syntax and could appear in programmatically generated stylesheets, especially when dealing with very small or large values. The current behavior silently converts these valid values to `"0pt"`, which could lead to unexpected rendering in styled dataframes.

Relevant documentation:
- W3C CSS Values and Units Module Level 3: https://www.w3.org/TR/css-values-3/
- W3C CSS Syntax Module Level 3: https://www.w3.org/TR/css-syntax-3/
- Source code location: `/pandas/io/formats/css.py:351`

## Proposed Fix

```diff
--- a/pandas/io/formats/css.py
+++ b/pandas/io/formats/css.py
@@ -348,7 +348,9 @@ class CSSResolver:
             )
             return self.size_to_pt("1!!default", conversions=conversions)

-        match = re.match(r"^(\S*?)([a-zA-Z%!].*)", in_val)
+        # Match numeric value (including scientific notation) followed by unit
+        # Captures full scientific notation format: optional sign, digits, optional decimal, optional exponent
+        match = re.match(r"^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)([a-zA-Z%!].*)", in_val)
         if match is None:
             return _error()
```