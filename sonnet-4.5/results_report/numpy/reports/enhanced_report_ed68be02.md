# Bug Report: numpy.char.capitalize Silently Truncates Unicode Strings That Expand During Capitalization

**Target**: `numpy.char.capitalize`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.char.capitalize` function silently truncates string data when Unicode characters expand in length during case conversion, violating its documented behavior and causing data loss without warning.

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=0x100, max_codepoint=0x1000, blacklist_categories=('Cs',)), min_size=1, max_size=10), min_size=1, max_size=10))
def test_capitalize_unicode(strings):
    arr = np.array(strings, dtype=str)
    result = char.capitalize(arr)

    for i in range(len(strings)):
        assert result[i] == strings[i].capitalize(), f"Failed on input: {strings[i]!r}, got {result[i]!r}, expected {strings[i].capitalize()!r}"

if __name__ == "__main__":
    test_capitalize_unicode()
```

<details>

<summary>
**Failing input**: `strings=['Āİ']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 14, in <module>
    test_capitalize_unicode()
    ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 6, in test_capitalize_unicode
    def test_capitalize_unicode(strings):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 11, in test_capitalize_unicode
    assert result[i] == strings[i].capitalize(), f"Failed on input: {strings[i]!r}, got {result[i]!r}, expected {strings[i].capitalize()!r}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Failed on input: 'Āİ', got np.str_('Āi'), expected 'Āi̇'
Falsifying example: test_capitalize_unicode(
    strings=['Āİ'],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.char as char

# Test the specific Unicode character mentioned in the bug report
arr = np.array(['ŉabc'], dtype=str)
result = char.capitalize(arr)

print(f"Input: {arr[0]!r}")
print(f"NumPy result: {result[0]!r}")
print(f"Python's str.capitalize result: {'ŉabc'.capitalize()!r}")
print(f"Input length: {len(arr[0])}")
print(f"NumPy result length: {len(result[0])}")
print(f"Expected length: {len('ŉabc'.capitalize())}")

# Verify the assertion fails
try:
    assert result[0] == 'ʼNabc'
    print("Assertion passed (unexpected)")
except AssertionError:
    print(f"Assertion failed: {result[0]!r} != 'ʼNabc'")
```

<details>

<summary>
Output demonstrating silent data truncation
</summary>
```
Input: np.str_('ŉabc')
NumPy result: np.str_('ʼNab')
Python's str.capitalize result: 'ʼNabc'
Input length: 4
NumPy result length: 4
Expected length: 5
Assertion failed: np.str_('ʼNab') != 'ʼNabc'
```
</details>

## Why This Is A Bug

This behavior violates the documented contract and causes data corruption in several ways:

1. **Documentation Contract Violation**: The numpy.char.capitalize documentation explicitly states it "calls str.capitalize() element-wise", creating a clear expectation that the behavior should match Python's str.capitalize(). However, the actual behavior differs significantly.

2. **Silent Data Loss**: When Unicode characters expand during capitalization (e.g., 'ŉ' (U+0149) becomes 'ʼN' (two characters: U+02BC + U+004E)), the function silently truncates the result to fit the original dtype size. In the example, 'ŉabc' should capitalize to 'ʼNabc' (5 characters), but numpy truncates it to 'ʼNab' (4 characters), losing the final 'c'.

3. **No Warning or Error**: The function provides no indication that data has been lost. Users may not realize their data has been corrupted until much later in their workflow.

4. **Inconsistent with Other numpy.char Functions**: Some other numpy.char functions like `add()` and `multiply()` properly handle dtype sizing for their output, creating an inconsistency in the API.

5. **Affects Multiple Unicode Characters**: This isn't limited to one obscure character. The hypothesis test quickly found another failing case with 'Āİ', and there are many Unicode characters that expand during case conversion.

## Relevant Context

The numpy.char module is marked as legacy in the NumPy documentation and users are encouraged to migrate to numpy.strings. However, this doesn't diminish the severity of silent data corruption for users still relying on this module.

The issue stems from NumPy's fixed-size string dtype system. When an array is created with dtype '<U4', it can only hold 4 Unicode characters. The capitalize operation doesn't check if the result will fit, nor does it allocate a properly sized output array.

Documentation references:
- numpy.char.capitalize: Claims to call str.capitalize() element-wise
- Python str.capitalize(): Handles Unicode expansion correctly with dynamic string sizing
- numpy.char module: Marked as legacy, but still widely used

## Proposed Fix

The capitalize function should calculate the maximum possible output length after case conversion and allocate an appropriately sized output array. Here's a high-level approach:

1. Before performing capitalization, iterate through all input strings to determine the maximum possible output length
2. Allocate the output array with sufficient size to hold the longest possible result
3. Perform the capitalization operation without truncation
4. Alternatively, at minimum, raise a warning when truncation occurs

Since the module is legacy, a simpler fix might be to add a clear warning in the documentation about this limitation and recommend using numpy.strings or pre-allocating arrays with sufficient size when working with Unicode text that may expand during case operations.