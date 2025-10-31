# Bug Report: numpy.strings.replace Single-Character Expansion Failure

**Target**: `numpy.strings.replace`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.strings.replace` function fails to correctly replace text when the entire input string consists only of the substring being replaced and the replacement is longer than the original.

## Property-Based Test

```python
import numpy as np
import numpy.strings as ns
from hypothesis import given, strategies as st, settings, assume


@given(
    st.lists(st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=0, max_size=20), min_size=1, max_size=20),
    st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=0, max_size=3),
    st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=0, max_size=3)
)
@settings(max_examples=500)
def test_replace_all_occurrences(string_list, old, new):
    assume(old != "")
    arr = np.array(string_list)
    result = ns.replace(arr, old, new)

    for i, s in enumerate(arr):
        expected = s.replace(old, new)
        actual = result[i]
        assert actual == expected, f"Failed for string '{s}': expected '{expected}', got '{actual}'"


if __name__ == "__main__":
    test_replace_all_occurrences()
```

<details>

<summary>
**Failing input**: `string_list=['a'], old='a', new='aa'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 24, in <module>
    test_replace_all_occurrences()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 7, in test_replace_all_occurrences
    st.lists(st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=0, max_size=20), min_size=1, max_size=20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 20, in test_replace_all_occurrences
    assert actual == expected, f"Failed for string '{s}': expected '{expected}', got '{actual}'"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Failed for string 'a': expected 'aa', got 'a'
Falsifying example: test_replace_all_occurrences(
    string_list=['a'],
    old='a',
    new='aa',
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as ns

# Test the bug case
arr = np.array(['a'])
result = ns.replace(arr, 'a', 'aa')

print("Input array:", arr)
print("Operation: replace 'a' with 'aa'")
print("Result:", result)
print("Expected:", ['aa'])
print("Match expected?", result[0] == 'aa')
print()

# Show that Python's native replace works correctly
print("Python's native str.replace:")
python_result = 'a'.replace('a', 'aa')
print(f"'a'.replace('a', 'aa') = '{python_result}'")
print()

# Test similar cases that work
print("Testing related cases:")
print("Case 1: 'ab' replace 'a' -> 'aa'")
arr2 = np.array(['ab'])
result2 = ns.replace(arr2, 'a', 'aa')
print(f"  Input: {arr2}, Result: {result2}, Expected: ['aab']")

print("Case 2: 'a' replace 'a' -> 'b' (same length)")
arr3 = np.array(['a'])
result3 = ns.replace(arr3, 'a', 'b')
print(f"  Input: {arr3}, Result: {result3}, Expected: ['b']")
```

<details>

<summary>
Output showing the bug: 'a' remains 'a' instead of becoming 'aa'
</summary>
```
Input array: ['a']
Operation: replace 'a' with 'aa'
Result: ['a']
Expected: ['aa']
Match expected? False

Python's native str.replace:
'a'.replace('a', 'aa') = 'aa'

Testing related cases:
Case 1: 'ab' replace 'a' -> 'aa'
  Input: ['ab'], Result: ['aab'], Expected: ['aab']
Case 2: 'a' replace 'a' -> 'b' (same length)
  Input: ['a'], Result: ['b'], Expected: ['b']
```
</details>

## Why This Is A Bug

This behavior violates the documented contract of `numpy.strings.replace`, which states it should return "a copy of the string with occurrences of substring old replaced by new." The function's documentation explicitly references `str.replace` as the behavioral model, establishing clear expectations.

The bug manifests specifically when:
1. The entire input string consists only of the substring being replaced (e.g., 'a' when replacing 'a')
2. The replacement string is longer than the original substring (e.g., 'aa' is longer than 'a')

This contradicts Python's native behavior where `'a'.replace('a', 'aa')` correctly returns `'aa'`. The inconsistency is particularly problematic because:
- The same operation works correctly when the string contains additional characters ('ab' → 'aab' works)
- Same-length replacements work correctly ('a' → 'b' works)
- The function silently returns incorrect results without any warning or error

## Relevant Context

- **NumPy Version**: 2.3.0
- **Documentation Reference**: The numpy.strings.replace documentation states "For each element in ``a``, return a copy of the string with occurrences of substring ``old`` replaced by ``new``" with a "See Also" reference to `str.replace`
- **Impact**: This affects data processing pipelines that rely on string manipulation, particularly those dealing with single-character tokens or symbols that need expansion
- **Workaround**: Users can detect and handle this case manually: `result = 'aa' if s == 'a' else ns.replace(s, 'a', 'aa')`

The bug appears to be a buffer allocation or length calculation issue in NumPy's C-level string implementation. When the entire string needs replacement with a longer string, the implementation likely:
1. Allocates a buffer based on the original string length
2. Fails to resize when the replacement would exceed this length
3. Returns the original string when the operation cannot be completed in the allocated space

## Proposed Fix

This is a complex C-level implementation issue that requires investigation of NumPy's string buffer management. A high-level approach to fixing this would involve:

1. **Proper length calculation**: Before allocating the result buffer, calculate the actual length needed by counting occurrences of the substring and computing: `new_length = original_length - (occurrences * old_length) + (occurrences * new_length)`

2. **Dynamic buffer allocation**: Ensure the result buffer is allocated with the correct size to accommodate the expanded string

3. **Edge case handling**: Add explicit handling for the case where the entire string is replaced, ensuring the buffer is properly sized even when `original_length == old_length`

4. **Testing**: Add regression tests specifically for:
   - Single character to multiple character replacement
   - Full string replacement with longer strings
   - Multiple occurrences in strings of various lengths

The fix would need to be implemented in NumPy's C string operations code, likely in the string replacement routine that handles the actual character copying and buffer management.