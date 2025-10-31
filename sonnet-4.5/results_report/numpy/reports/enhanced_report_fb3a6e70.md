# Bug Report: numpy.char Case Functions Silently Truncate Unicode Expansions

**Target**: `numpy.char.upper()`, `numpy.char.capitalize()`, `numpy.char.swapcase()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

numpy.char case conversion functions silently truncate Unicode characters that expand during case mapping (e.g., 'ß' → 'SS'), causing data corruption without any warning or error.

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=0, max_size=100))
@settings(max_examples=500, deadline=None)
def test_upper_matches_python(s):
    numpy_result = char.upper(s)
    numpy_str = str(numpy_result) if hasattr(numpy_result, 'item') else numpy_result
    python_result = s.upper()
    assert numpy_str == python_result

if __name__ == "__main__":
    test_upper_matches_python()
```

<details>

<summary>
**Failing input**: `s='ß'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 13, in <module>
    test_upper_matches_python()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 5, in test_upper_matches_python
    @settings(max_examples=500, deadline=None)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 10, in test_upper_matches_python
    assert numpy_str == python_result
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_upper_matches_python(
    s='ß',
)
```
</details>

## Reproducing the Bug

```python
import numpy.char as char

test_cases = [
    ('ß', 'upper'),
    ('straße', 'upper'),
    ('ﬁ', 'upper'),
    ('ß', 'capitalize'),
    ('ß', 'swapcase'),
]

for s, func_name in test_cases:
    numpy_func = getattr(char, func_name)
    python_func = getattr(str, func_name)

    numpy_result = numpy_func(s).item()
    python_result = python_func(s)

    print(f"{func_name}({repr(s)}):")
    print(f"  numpy:  {repr(numpy_result)} (length {len(numpy_result)})")
    print(f"  python: {repr(python_result)} (length {len(python_result)})")

    if numpy_result != python_result:
        print(f"  TRUNCATED!")
    print()
```

<details>

<summary>
Silent truncation in all tested case conversions
</summary>
```
upper('ß'):
  numpy:  'S' (length 1)
  python: 'SS' (length 2)
  TRUNCATED!

upper('straße'):
  numpy:  'STRASS' (length 6)
  python: 'STRASSE' (length 7)
  TRUNCATED!

upper('ﬁ'):
  numpy:  'F' (length 1)
  python: 'FI' (length 2)
  TRUNCATED!

capitalize('ß'):
  numpy:  'S' (length 1)
  python: 'Ss' (length 2)
  TRUNCATED!

swapcase('ß'):
  numpy:  'S' (length 1)
  python: 'SS' (length 2)
  TRUNCATED!

```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **Direct contradiction of documentation**: The numpy.char.upper() documentation explicitly states it "Calls :meth:`str.upper` element-wise", yet `char.upper('ß')` returns 'S' while `str.upper('ß')` returns 'SS'. The documentation makes no mention of truncation limitations.

2. **Silent data corruption**: The functions truncate expanded characters without raising any warning or error. Users processing German text get incorrect results ('S' instead of 'SS') with no indication that data has been lost.

3. **Violates Unicode standards**: The Unicode Standard (Technical Report #21) explicitly defines that U+00DF 'ß' (small letter sharp s) expands to 'SS' when uppercased. This is not an edge case but a well-defined mapping in the Unicode SpecialCasing file.

4. **Breaks case-insensitive operations**: Case-insensitive string matching relies on correct case conversion. When `upper('ß')` returns 'S' instead of 'SS', case-insensitive comparisons will fail for German text.

5. **Fixed-size dtype limitation not documented**: NumPy maintains the original dtype size (e.g., '<U1' for single character) even when the case conversion expands the string, causing truncation to fit the original size constraint.

## Relevant Context

The issue stems from NumPy's use of fixed-size character arrays. When you create an array with `np.array(['ß'])`, it gets dtype '<U1' (single Unicode character). The upper() function maintains this dtype, truncating the two-character result 'SS' to 'S'.

Affected Unicode characters include:
- German sharp s (ß → SS)
- Ligatures (ﬁ → FI, ﬂ → FL)
- Various Greek and other language-specific characters with expanding case mappings

This affects real-world applications processing:
- German text (common in scientific computing)
- Typography and publishing systems
- Multilingual text processing
- Database operations requiring case-insensitive matching

NumPy version tested: 2.3.0

## Proposed Fix

The issue requires handling expanding case mappings in NumPy's string operations. A high-level approach would involve:

1. **Pre-scan phase**: Before case conversion, scan input to determine if any characters will expand (check against Unicode SpecialCasing data)

2. **Dynamic dtype allocation**: If expansion is needed, allocate output array with sufficient size (e.g., '<U2' or larger as needed)

3. **Alternative: Raise warning/error**: If maintaining fixed-size is critical, at minimum raise a warning when truncation will occur:
   ```python
   UserWarning: Unicode case conversion would expand 'ß' to 'SS' but output is truncated to fit dtype '<U1>'
   ```

4. **Documentation update**: If the limitation cannot be fixed, explicitly document that these functions do not support Unicode characters that expand during case conversion and will silently truncate results.

The proper fix requires changes to NumPy's core string ufunc implementations to handle variable-length outputs or at least detect and warn about truncation scenarios.