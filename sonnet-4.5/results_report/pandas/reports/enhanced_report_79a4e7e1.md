# Bug Report: numpy.char.upper() Truncates Unicode Characters During Case Expansion

**Target**: `numpy.char.upper()`, `numpy.char.swapcase()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.upper()` and `numpy.char.swapcase()` silently truncate Unicode characters that expand in length during case conversion (e.g., German sharp s 'ß' → 'SS'), causing data loss without warning.

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=0, max_size=100))
@settings(max_examples=500, deadline=None)
def test_swapcase_involution(s):
    swap1 = char.swapcase(s)
    swap2 = char.swapcase(swap1)
    swap2_str = str(swap2) if hasattr(swap2, 'item') else swap2
    assert swap2_str == s

if __name__ == "__main__":
    test_swapcase_involution()
```

<details>

<summary>
**Failing input**: `s='ß'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 13, in <module>
    test_swapcase_involution()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 5, in test_swapcase_involution
    @settings(max_examples=500, deadline=None)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 10, in test_swapcase_involution
    assert swap2_str == s
           ^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_swapcase_involution(
    s='ß',
)
```
</details>

## Reproducing the Bug

```python
import numpy.char as char

s = 'ß'

numpy_upper = char.upper(s).item()
python_upper = s.upper()

print(f"Input: {repr(s)}")
print(f"numpy.char.upper():  {repr(numpy_upper)} (length {len(numpy_upper)})")
print(f"Python str.upper():  {repr(python_upper)} (length {len(python_upper)})")
print()

# Test with swapcase
swap1 = char.swapcase(s)
swap2 = char.swapcase(swap1)
swap2_str = str(swap2) if hasattr(swap2, 'item') else swap2

print("Swapcase test:")
print(f"Original: {repr(s)}")
print(f"After first swapcase: {repr(str(swap1))}")
print(f"After second swapcase: {repr(swap2_str)}")
print(f"Should be equal to original? {swap2_str == s}")
print()

# Verify that this is a data corruption issue
print("Assertion test:")
try:
    assert numpy_upper == python_upper, f"numpy.char.upper() returned {repr(numpy_upper)}, but Python str.upper() returned {repr(python_upper)}"
    print("✓ numpy.char.upper() matches Python str.upper()")
except AssertionError as e:
    print(f"✗ FAILED: {e}")
```

<details>

<summary>
Data truncation: 'ß' becomes 'S' instead of 'SS'
</summary>
```
Input: 'ß'
numpy.char.upper():  'S' (length 1)
Python str.upper():  'SS' (length 2)

Swapcase test:
Original: 'ß'
After first swapcase: 'S'
After second swapcase: 's'
Should be equal to original? False

Assertion test:
✗ FAILED: numpy.char.upper() returned 'S', but Python str.upper() returned 'SS'
```
</details>

## Why This Is A Bug

1. **Silent data corruption**: The function silently truncates the result when Unicode case conversion expands the string length. The German sharp s 'ß' should become 'SS' (two characters) but is truncated to just 'S' (one character), losing data without any warning or error.

2. **Violates documented behavior**: The numpy.char.upper() documentation explicitly states "Calls :meth:`str.upper` element-wise" and references Python's str.upper() as the behavior model. However, the actual output differs: Python's str.upper('ß') correctly returns 'SS', while numpy.char.upper('ß') returns only 'S'.

3. **Breaks mathematical properties**: The swapcase operation should be an involution for single-case strings (swapcase(swapcase(x)) == x), but this property is violated. swapcase('ß') returns 'S', and swapcase('S') returns 's', not 'ß'.

4. **Affects real-world text**: This affects legitimate German text containing common words like "Straße" (street), "Fuß" (foot), "groß" (large), and "schließen" (close). Testing shows that the text "Straße Fuß groß heißen schließen Spaß" loses 6 characters when processed.

5. **Inconsistent behavior in arrays**: When using NumPy arrays with dtype '<U9', some words work correctly while others like "schließen" still get truncated to "SCHLIESSE" (missing the final 'N'), making the bug unpredictable and hard to detect.

## Relevant Context

The bug stems from NumPy's use of fixed-size Unicode arrays (dtype '<U#'). When a character like 'ß' is stored in a '<U1' array and uppercase is applied, the result 'SS' requires '<U2' but the array cannot expand, causing truncation.

This affects multiple Unicode ligatures and special characters:
- 'ß' (U+00DF) → 'SS' (loses 1 character)
- 'ﬀ' (U+FB00) → 'FF' (loses 1 character)
- 'ﬁ' (U+FB01) → 'FI' (loses 1 character)
- 'ﬂ' (U+FB02) → 'FL' (loses 1 character)

NumPy documentation: https://numpy.org/doc/stable/reference/generated/numpy.char.upper.html
Unicode case mappings: https://www.unicode.org/Public/UCD/latest/ucd/SpecialCasing.txt

## Proposed Fix

The issue requires handling Unicode expansion in case operations. Since NumPy uses fixed-size arrays and cannot dynamically resize, there are three possible approaches:

1. **Pre-calculate expanded size**: Before performing case conversion, scan the input to determine the maximum expanded size needed, then allocate the output array accordingly.

2. **Raise an error on truncation**: Detect when truncation would occur and raise a ValueError with a clear message instead of silently corrupting data.

3. **Document the limitation**: If the behavior cannot be changed due to architectural constraints, the documentation should prominently warn about this limitation and provide examples of affected characters.

A minimal documentation fix would be:

```diff
 def upper(a):
     """
     Return an array with the elements converted to uppercase.

     Calls :meth:`str.upper` element-wise.
+
+    .. warning::
+        Characters that expand during case conversion (e.g., 'ß' → 'SS')
+        will be truncated to fit the original array size. This affects
+        certain Unicode ligatures and special characters. For accurate
+        case conversion of such text, use Python's str.upper() directly.

     For 8-bit strings, this method is locale-dependent.
```