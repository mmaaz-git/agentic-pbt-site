# Bug Report: numpy.char.replace Incorrectly Handles Null Bytes and Other Edge Cases

**Target**: `numpy.char.replace`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.replace()` has critical bugs when handling null bytes (`\x00`) and certain replacement patterns, producing incorrect results that violate Python's `str.replace()` behavior and cause data corruption.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for numpy.char.replace"""

import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=0, max_size=20), st.text(min_size=1, max_size=5), st.text(min_size=0, max_size=5))
@settings(max_examples=1000)
def test_replace_matches_python(s, old, new):
    py_result = s.replace(old, new)
    np_result = str(char.replace(s, old, new))

    assert py_result == np_result, f"replace({repr(s)}, {repr(old)}, {repr(new)}): Python={repr(py_result)}, NumPy={repr(np_result)}"

if __name__ == "__main__":
    # Run the test
    test_replace_matches_python()
```

<details>

<summary>
**Failing input**: `s='0'`, `old='0'`, `new='00'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 17, in <module>
    test_replace_matches_python()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 8, in test_replace_matches_python
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 13, in test_replace_matches_python
    assert py_result == np_result, f"replace({repr(s)}, {repr(old)}, {repr(new)}): Python={repr(py_result)}, NumPy={repr(np_result)}"
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: replace('0', '0', '00'): Python='00', NumPy='0'
Falsifying example: test_replace_matches_python(
    s='0',
    old='0',
    new='00',
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Reproduction of numpy.char.replace null byte bug"""

import numpy.char as char

print("=== NumPy char.replace Null Byte Bug Reproduction ===\n")

print("Bug 1: Null byte as 'old' parameter when string doesn't contain null bytes")
print("-" * 70)
s = 'test'
py_result = s.replace('\x00', 'X')
np_result = str(char.replace(s, '\x00', 'X'))
print(f"Input: char.replace('test', '\\x00', 'X')")
print(f"Expected (Python): {repr(py_result)}")
print(f"Actual (NumPy):    {repr(np_result)}")
print(f"Match: {py_result == np_result}\n")

print("Bug 2: Removing null bytes from string")
print("-" * 70)
s = 'te\x00st'
py_result = s.replace('\x00', '')
np_result = str(char.replace(s, '\x00', ''))
print(f"Input: char.replace('te\\x00st', '\\x00', '')")
print(f"Expected (Python): {repr(py_result)}")
print(f"Actual (NumPy):    {repr(np_result)}")
print(f"Match: {py_result == np_result}\n")

print("Bug 3: Replacing with null bytes")
print("-" * 70)
s = 'test'
py_result = s.replace('e', '\x00')
np_result = str(char.replace(s, 'e', '\x00'))
print(f"Input: char.replace('test', 'e', '\\x00')")
print(f"Expected (Python): {repr(py_result)}")
print(f"Actual (NumPy):    {repr(np_result)}")
print(f"Match: {py_result == np_result}\n")

print("Additional test: Empty string vs null byte")
print("-" * 70)
s = 'test'
empty_result = str(char.replace(s, '', 'X'))
null_result = str(char.replace(s, '\x00', 'X'))
print(f"char.replace('test', '', 'X') = {repr(empty_result)}")
print(f"char.replace('test', '\\x00', 'X') = {repr(null_result)}")
print(f"Same result: {empty_result == null_result}\n")

print("Complex case: Multiple null bytes")
print("-" * 70)
s = 'a\x00b\x00c'
py_result = s.replace('\x00', '-')
np_result = str(char.replace(s, '\x00', '-'))
print(f"Input: char.replace('a\\x00b\\x00c', '\\x00', '-')")
print(f"Expected (Python): {repr(py_result)}")
print(f"Actual (NumPy):    {repr(np_result)}")
print(f"Match: {py_result == np_result}")
```

<details>

<summary>
Multiple critical bugs in numpy.char.replace demonstrated
</summary>
```
=== NumPy char.replace Null Byte Bug Reproduction ===

Bug 1: Null byte as 'old' parameter when string doesn't contain null bytes
----------------------------------------------------------------------
Input: char.replace('test', '\x00', 'X')
Expected (Python): 'test'
Actual (NumPy):    'XtXeXsXtX'
Match: False

Bug 2: Removing null bytes from string
----------------------------------------------------------------------
Input: char.replace('te\x00st', '\x00', '')
Expected (Python): 'test'
Actual (NumPy):    'te\x00st'
Match: False

Bug 3: Replacing with null bytes
----------------------------------------------------------------------
Input: char.replace('test', 'e', '\x00')
Expected (Python): 't\x00st'
Actual (NumPy):    'tst'
Match: False

Additional test: Empty string vs null byte
----------------------------------------------------------------------
char.replace('test', '', 'X') = 'XtXeXsXtX'
char.replace('test', '\x00', 'X') = 'XtXeXsXtX'
Same result: True

Complex case: Multiple null bytes
----------------------------------------------------------------------
Input: char.replace('a\x00b\x00c', '\x00', '-')
Expected (Python): 'a-b-c'
Actual (NumPy):    '-a-\x00-b-\x00-c-'
Match: False
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **Null byte mishandled as empty string**: When `\x00` is used as the search pattern, NumPy treats it identically to an empty string `''`, causing it to insert replacements between every character. This is objectively wrong - null bytes should be treated as regular characters.

2. **Cannot detect or remove actual null bytes**: When strings contain actual null bytes, the function fails to find and replace them. This makes it impossible to process binary data correctly.

3. **Strips null bytes from replacements**: When null bytes appear in the `new` parameter, they are silently removed from the output string instead of being inserted.

4. **Buffer size calculation errors**: The Hypothesis test reveals that even without null bytes, certain replacements fail (e.g., replacing '0' with '00' results in '0' instead of '00'), indicating fundamental issues with buffer size calculation.

5. **Violates documented contract**: The function's docstring states it performs replacement "similarly to Python's standard str.replace() method", but the behavior is completely different, causing silent data corruption.

## Relevant Context

The implementation is in `numpy._core.umath._replace`, which is a C extension. The Python wrapper in `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/_core/strings.py:1307` performs buffer size calculations before calling the C function.

Key observations from the code:
- Line 1361-1363: Buffer size is calculated based on counts of replacements
- Line 1367: The `_replace` C function is called with pre-allocated output buffer
- The C implementation appears to use null-terminated string operations, causing the null byte issues

This bug affects all code that:
- Processes binary data containing null bytes
- Works with network protocols or file formats that use null bytes
- Attempts to clean or validate string data
- Performs text transformations where the replacement might expand the string

## Proposed Fix

The fix requires changes to the C implementation of `_replace` in `numpy._core.umath`:

1. Use length-aware string operations instead of null-terminated C string functions
2. Properly handle null bytes as regular characters, not string terminators
3. Fix buffer size calculations to handle expanding replacements correctly
4. Add comprehensive tests for null byte handling and edge cases

Since the C code is not directly accessible in this environment, here's the high-level approach:

- Replace all `strlen()` calls with length-tracked operations
- Use `memcpy()` or `memmove()` instead of `strcpy()` for string operations
- Track string lengths explicitly rather than relying on null termination
- Ensure the buffer size calculation accounts for all replacement scenarios
- Add specific handling for the case where `old` is a null byte