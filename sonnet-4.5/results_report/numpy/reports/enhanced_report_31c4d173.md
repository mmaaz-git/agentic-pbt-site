# Bug Report: numpy.char Case Transformation Functions Silently Truncate Multi-Character Unicode Results

**Target**: `numpy.char.upper`, `numpy.char.swapcase`, `numpy.char.capitalize`, `numpy.char.title`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

NumPy's char case transformation functions silently truncate results when Unicode case transformations produce multi-character outputs, violating their documented contract to call Python's corresponding str methods element-wise.

## Property-Based Test

```python
import numpy as np
import numpy.char
from hypothesis import given, strategies as st


@given(st.lists(st.text(), min_size=1))
def test_swapcase_matches_python(strings):
    arr = np.array(strings)
    numpy_result = numpy.char.swapcase(arr)

    for i in range(len(strings)):
        python_result = strings[i].swapcase()
        assert numpy_result[i] == python_result
```

<details>

<summary>
**Failing input**: `strings=['ß']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 36, in <module>
    test_swapcase_matches_python()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 7, in test_swapcase_matches_python
    def test_swapcase_matches_python(strings):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 13, in test_swapcase_matches_python
    assert numpy_result[i] == python_result
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_swapcase_matches_python(
    strings=['ß'],
)
Test failed for input: ['ẖ']
NumPy result: 'H'
Python result: 'H̱'
NumPy == Python: False

Running property-based test...
Property-based test found failure
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.char

# Test character that demonstrates the bug
test_char = 'ẖ'
arr = np.array([test_char])

# Get results from NumPy and Python
numpy_result = numpy.char.swapcase(arr)[0]
python_result = test_char.swapcase()

print(f"Input: '{test_char}' (U+{ord(test_char):04X}, LATIN SMALL LETTER H WITH LINE BELOW)")
print(f"Python str.swapcase: '{python_result}' (len={len(python_result)})")
print(f"NumPy char.swapcase: '{numpy_result}' (len={len(numpy_result)})")
print(f"Match: {str(numpy_result) == python_result}")

# Show the character codes to demonstrate truncation
print(f"\nPython result characters: {[f'U+{ord(c):04X}' for c in python_result]}")
print(f"NumPy result characters: {[f'U+{ord(c):04X}' for c in str(numpy_result)]}")

print("\n--- Testing other affected functions and characters ---")

# Test multiple functions with different problematic characters
test_cases = [
    ('ẖ', 'LATIN SMALL LETTER H WITH LINE BELOW'),
    ('ǰ', 'LATIN SMALL LETTER J WITH CARON'),
    ('ß', 'LATIN SMALL LETTER SHARP S (German eszett)')
]

functions = [
    ('upper', numpy.char.upper),
    ('swapcase', numpy.char.swapcase),
    ('capitalize', numpy.char.capitalize),
    ('title', numpy.char.title)
]

for char, description in test_cases:
    print(f"\nCharacter: '{char}' (U+{ord(char):04X}, {description})")
    for func_name, numpy_func in functions:
        arr = np.array([char])
        numpy_res = str(numpy_func(arr)[0])
        python_res = getattr(char, func_name)()
        match = "✓" if numpy_res == python_res else "✗"
        print(f"  {func_name:10} → NumPy: {repr(numpy_res):8} Python: {repr(python_res):8} {match}")
```

<details>

<summary>
Output demonstrating silent truncation
</summary>
```
Input: 'ẖ' (U+1E96, LATIN SMALL LETTER H WITH LINE BELOW)
Python str.swapcase: 'H̱' (len=2)
NumPy char.swapcase: 'H' (len=1)
Match: False

Python result characters: ['U+0048', 'U+0331']
NumPy result characters: ['U+0048']

--- Testing other affected functions and characters ---

Character: 'ẖ' (U+1E96, LATIN SMALL LETTER H WITH LINE BELOW)
  upper      → NumPy: 'H'      Python: 'H̱'     ✗
  swapcase   → NumPy: 'H'      Python: 'H̱'     ✗
  capitalize → NumPy: 'H'      Python: 'H̱'     ✗
  title      → NumPy: 'H'      Python: 'H̱'     ✗

Character: 'ǰ' (U+01F0, LATIN SMALL LETTER J WITH CARON)
  upper      → NumPy: 'J'      Python: 'J̌'     ✗
  swapcase   → NumPy: 'J'      Python: 'J̌'     ✗
  capitalize → NumPy: 'J'      Python: 'J̌'     ✗
  title      → NumPy: 'J'      Python: 'J̌'     ✗

Character: 'ß' (U+00DF, LATIN SMALL LETTER SHARP S (German eszett))
  upper      → NumPy: 'S'      Python: 'SS'     ✗
  swapcase   → NumPy: 'S'      Python: 'SS'     ✗
  capitalize → NumPy: 'S'      Python: 'Ss'     ✗
  title      → NumPy: 'S'      Python: 'Ss'     ✗
```
</details>

## Why This Is A Bug

This bug violates the explicit documented contract of these NumPy functions. Each function's documentation states it "Calls str.[method] element-wise", creating a clear expectation that results will match Python's built-in string methods. However:

1. **Silent data loss**: When Unicode case transformations expand from one character to multiple characters (either through decomposition with combining diacritics like 'ẖ' → 'H̱' or actual expansion like 'ß' → 'SS'), NumPy silently truncates to the first character only. The combining diacritics (U+0331 for 'H̱', U+030C for 'J̌') are completely lost.

2. **Dtype width limitation**: NumPy preserves the input array's dtype width (e.g., '<U1' for single-character strings) even when the correct result requires more characters. The functions don't allocate wider dtypes to accommodate expanded results.

3. **No warning or error**: The truncation happens silently with no indication to users that data has been lost, making this particularly dangerous for data processing pipelines.

4. **Affects real languages**: This impacts German text (ß → SS), multiple Latin-script languages using combining diacritics, and other Unicode normalization cases. These aren't edge cases but legitimate text processing scenarios.

5. **All case functions affected**: The bug consistently affects `upper()`, `swapcase()`, `capitalize()`, and `title()` - all core text transformation functions that users rely on for data preprocessing.

## Relevant Context

The root cause appears to be NumPy's fixed-width string dtype system. When creating an array with `np.array(['ß'])`, NumPy allocates a dtype of '<U1' (1 Unicode character). The case transformation functions preserve this dtype width even when the correct result requires more space, leading to truncation.

NumPy documentation for these functions can be found at:
- https://numpy.org/doc/stable/reference/generated/numpy.char.upper.html
- https://numpy.org/doc/stable/reference/generated/numpy.char.swapcase.html

The Unicode characters demonstrating this issue are legitimate and commonly used:
- U+00DF (ß): German sharp s, uppercases to 'SS'
- U+1E96 (ẖ): Used in transliteration systems, uppercases to H + combining macron below (U+0048 + U+0331)
- U+01F0 (ǰ): Used in various Latin-script languages, uppercases to J + combining caron (U+004A + U+030C)

## Proposed Fix

Due to NumPy's architectural constraint of fixed-width dtypes, a complete fix is non-trivial. The functions would need to:
1. Pre-scan all strings to determine the maximum character width after transformation
2. Allocate a new array with appropriate dtype width
3. Apply the transformations without truncation

A minimal interim fix should at minimum warn users when truncation occurs and update documentation to clarify this limitation. Here's a conceptual approach for adding warnings:

```diff
# Conceptual patch for numpy.char functions
def upper(a):
    """
    Return an array with the elements converted to uppercase.

    Calls str.upper element-wise.

+   Warning
+   -------
+   When Unicode case transformations produce multi-character results
+   (e.g., German ß → SS), the output may be truncated to fit the
+   original dtype width. For full Unicode support, consider using
+   Python's str.upper() directly before converting to NumPy arrays.

    """
    result = _vec_string(a, np.str_, 'upper')
+   # Check if truncation occurred
+   for i, (orig, res) in enumerate(zip(a.flat, result.flat)):
+       expected = str(orig).upper()
+       if str(res) != expected:
+           import warnings
+           warnings.warn(
+               f"Case transformation truncated: '{orig}' → '{res}' "
+               f"(expected '{expected}'). Consider using wider dtype.",
+               UnicodeWarning,
+               stacklevel=2
+           )
+           break  # Warn only once per operation
    return result
```