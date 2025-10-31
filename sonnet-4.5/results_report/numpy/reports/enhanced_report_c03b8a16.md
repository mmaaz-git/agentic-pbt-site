# Bug Report: numpy.char.title Silently Truncates Unicode Characters That Expand During Case Conversion

**Target**: `numpy.char.title`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.title` silently truncates string data when Unicode characters expand during title case conversion, causing data loss without warning or error.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis test that discovered the numpy.char.title truncation bug
with Unicode characters that expand during case conversion.
"""

import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=0x100, max_codepoint=0x1000, blacklist_categories=('Cs',)), min_size=1, max_size=10), min_size=1, max_size=10))
def test_title_unicode(strings):
    arr = np.array(strings, dtype=str)
    result = char.title(arr)

    for i in range(len(strings)):
        assert result[i] == strings[i].title()

if __name__ == "__main__":
    test_title_unicode()
```

<details>

<summary>
**Failing input**: `strings=['Āİ']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/numpy_hypo.py", line 20, in <module>
    test_title_unicode()
    ~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/36/numpy_hypo.py", line 12, in test_title_unicode
    def test_title_unicode(strings):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/36/numpy_hypo.py", line 17, in test_title_unicode
    assert result[i] == strings[i].title()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_title_unicode(
    strings=['Āİ'],
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction case demonstrating numpy.char.title truncation bug
with Unicode ligatures that expand during case conversion.
"""

import numpy as np
import numpy.char as char

# Test case with the ligature 'ﬁ' (U+FB01) which title-cases to 'Fi' (2 chars)
arr = np.array(['ﬁ test'], dtype=str)
result = char.title(arr)

print(f"Input:    {arr[0]!r}")
print(f"Result:   {result[0]!r}")
print(f"Expected: {'ﬁ test'.title()!r}")

print(f"\nInput dtype:  {arr.dtype}")
print(f"Result dtype: {result.dtype}")

print(f"\nInput length:    {len(arr[0])}")
print(f"Result length:   {len(result[0])}")
print(f"Expected length: {len('ﬁ test'.title())}")

# This assertion will fail due to truncation
assert result[0] == 'ﬁ test'.title(), f"Got {result[0]!r} but expected {'ﬁ test'.title()!r}"
```

<details>

<summary>
Truncated output: 'Fi Tes' instead of 'Fi Test'
</summary>
```
Input:    np.str_('ﬁ test')
Result:   np.str_('Fi Tes')
Expected: 'Fi Test'

Input dtype:  <U6
Result dtype: <U6

Input length:    6
Result length:   6
Expected length: 7
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/numpy_repo.py", line 26, in <module>
    assert result[0] == 'ﬁ test'.title(), f"Got {result[0]!r} but expected {'ﬁ test'.title()!r}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Got np.str_('Fi Tes') but expected 'Fi Test'
```
</details>

## Why This Is A Bug

The numpy.char.title function violates its documented contract in multiple critical ways:

1. **Documentation Violation**: The function documentation explicitly states it "Calls :meth:`str.title` element-wise" (numpy/_core/strings.py:1267), but it produces different results when Unicode expansion occurs. The ligature 'ﬁ' (U+FB01) correctly title-cases to 'Fi' (2 characters) in Python's str.title(), but numpy truncates the result.

2. **Silent Data Corruption**: The function silently truncates data without warning when the title-cased result exceeds the input dtype size. The input 'ﬁ test' (6 characters, dtype='<U6') expands to 'Fi Test' (7 characters), but numpy returns 'Fi Tes' (6 characters), losing the final 't'.

3. **Inconsistent with Other numpy.char Functions**: Functions like `char.add()` and `char.multiply()` correctly handle dtype expansion when concatenating or repeating strings, automatically allocating larger output arrays. The title() function should similarly calculate the required output size.

4. **Principle of Least Surprise Violation**: Users reasonably expect identical results to Python's str.title() based on the documentation, especially since other numpy.char functions handle expansion correctly.

## Relevant Context

The implementation is located at `/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/strings.py:1260-1292`. The function uses `_vec_string(a_arr, a_arr.dtype, 'title')` which preserves the input dtype instead of calculating the maximum possible expansion.

Unicode characters that can expand during title-casing include:
- Ligatures: 'ﬁ' (U+FB01) → 'Fi', 'ﬂ' (U+FB02) → 'Fl'
- Turkish İ (U+0130) which can expand in certain contexts
- Various other Unicode normalization cases

Documentation: https://numpy.org/doc/stable/reference/generated/numpy.char.title.html
Source: https://github.com/numpy/numpy/blob/main/numpy/_core/strings.py

## Proposed Fix

The title() function needs to calculate the maximum possible output size after Unicode expansion, similar to how char.add() handles dtype sizing. Here's a high-level approach:

1. Pre-scan the input array to determine the maximum expanded length after title-casing
2. Allocate an output array with appropriate dtype size
3. Apply the title case transformation to the correctly-sized array

This would require modifying the C implementation in `_vec_string` to handle dynamic output sizing for case transformation operations, or implementing title() as a proper ufunc (which is already planned according to code comments at numpy/_core/strings.py:98-99).