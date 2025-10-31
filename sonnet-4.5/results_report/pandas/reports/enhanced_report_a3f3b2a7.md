# Bug Report: pandas.io.formats.format._trim_zeros_complex Removes Closing Parenthesis from Complex Numbers

**Target**: `pandas.io.formats.format._trim_zeros_complex`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_trim_zeros_complex` function incorrectly removes the closing parenthesis from complex number string representations, producing malformed strings like `(1+2j` instead of the correct `(1+2j)`.

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


if __name__ == "__main__":
    # Run the test
    test_trim_zeros_complex_preserves_parentheses()
```

<details>

<summary>
**Failing input**: `[(1.0, 1.0)]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 21, in <module>
    test_trim_zeros_complex_preserves_parentheses()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 6, in test_trim_zeros_complex_preserves_parentheses
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 16, in test_trim_zeros_complex_preserves_parentheses
    assert result.endswith(')'), f"Lost closing parenthesis: {original} -> {result}"
           ~~~~~~~~~~~~~~~^^^^^
AssertionError: Lost closing parenthesis: (1+1j) -> (1+1j
Falsifying example: test_trim_zeros_complex_preserves_parentheses(
    float_pairs=[(1.0, 1.0)],
)
```
</details>

## Reproducing the Bug

```python
from pandas.io.formats.format import _trim_zeros_complex

# Test with complex numbers that have both real and imaginary parts
values = [complex(1, 2), complex(3, 4)]
str_values = [str(v) for v in values]

print(f"Input:  {str_values}")
result = _trim_zeros_complex(str_values)
print(f"Output: {result}")

# Let's also test with a single value to be more explicit
single_value = complex(1.0, 1.0)
single_str = str(single_value)
print(f"\nSingle Input:  '{single_str}'")
single_result = _trim_zeros_complex([single_str])
print(f"Single Output: '{single_result[0]}'")

# Check if parenthesis was lost
if single_str.endswith(')') and not single_result[0].endswith(')'):
    print("\nERROR: Closing parenthesis was removed!")
    print(f"  Expected: '{single_str}'")
    print(f"  Got:      '{single_result[0]}'")
```

<details>

<summary>
Output shows malformed complex number strings with missing closing parenthesis
</summary>
```
Input:  ['(1+2j)', '(3+4j)']
Output: ['(1+2j', '(3+4j']

Single Input:  '(1+1j)'
Single Output: '(1+1j'

ERROR: Closing parenthesis was removed!
  Expected: '(1+1j)'
  Got:      '(1+1j'
```
</details>

## Why This Is A Bug

This violates Python's standard complex number string representation format. According to Python's documentation, complex numbers with non-zero real parts are represented with parentheses: `(real+imagj)`. The function produces syntactically invalid output that:

1. **Breaks Python's standard format**: The string `(1+2j` is not a valid Python expression and cannot be parsed back to a complex number using `complex()` or `eval()`.

2. **Violates the function's purpose**: The function is named `_trim_zeros_complex` which implies it should only trim trailing zeros from numeric parts while preserving the structural format. The docstring states it "executes the _trim_zeros_float method on each of those [parts]" - not that it modifies the overall structure.

3. **Creates malformed display output**: This function is used by pandas' FloatArrayFormatter when `fixed_width=True`, meaning DataFrames will display malformed complex numbers to users.

4. **Inconsistent parsing logic**: The regex split `re.split(r"([j+-])", x)` on input `"(1+2j)"` produces `['(1', '+', '2', 'j', ')']`. The function then takes:
   - `trimmed[:-4]` for real part → `'(1'` (missing closing paren)
   - `trimmed[-4:-2]` for imaginary part → `'+2'`
   - The closing `)` at index -1 is never captured

## Relevant Context

- **Function location**: `/pandas/io/formats/format.py` lines 1760-1790
- **Python complex number specification**: https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex
- **Usage context**: Called by FloatArrayFormatter when formatting complex arrays with fixed width
- **Python behavior**: `str(complex(1, 2))` always produces `'(1+2j)'` with parentheses when real part is non-zero
- **Impact**: Affects all pandas DataFrame displays containing complex numbers when using fixed-width formatting

The bug occurs because the function incorrectly assumes the closing parenthesis is part of the imaginary component parsing, but the regex split isolates it as a separate element that gets discarded.

## Proposed Fix

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1764,11 +1764,18 @@ def _trim_zeros_complex(str_complexes: ArrayLike, decimal: str = ".") -> list[s
     executes the _trim_zeros_float method on each of those.
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
         # and the real part is everything before the imaginary part
         trimmed = re.split(r"([j+-])", x)
         real_part.append("".join(trimmed[:-4]))
         imag_part.append("".join(trimmed[-4:-2]))
@@ -1780,13 +1787,16 @@ def _trim_zeros_complex(str_complexes: ArrayLike, decimal: str = ".") -> list[s
     padded_parts = _trim_zeros_float(real_part + imag_part, decimal)
     if len(padded_parts) == 0:
         return []
     padded_length = max(len(part) for part in padded_parts) - 1
     padded = [
         real_pt  # real part, possibly NaN
         + imag_pt[0]  # +/-
         + f"{imag_pt[1:]:>{padded_length}}"  # complex part (no sign), possibly nan
         + "j"
-        for real_pt, imag_pt in zip(padded_parts[:n], padded_parts[n:])
+        for i, (real_pt, imag_pt) in enumerate(zip(padded_parts[:n], padded_parts[n:]))
     ]
+    # Add back parentheses where they were present
+    padded = [f"({p})" if has_parens[i] else p for i, p in enumerate(padded)]
+
     return padded
```