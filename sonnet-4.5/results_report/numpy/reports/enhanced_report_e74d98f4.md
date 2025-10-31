# Bug Report: numpy.char.multiply Silently Strips Trailing Null Bytes

**Target**: `numpy.char.multiply`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.multiply` silently strips trailing null bytes from strings during multiplication, resulting in data loss without warning and behavior that differs from Python's native string multiplication operator.

## Property-Based Test

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env')

import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st, settings


@st.composite
def text_ending_with_null(draw):
    prefix = draw(st.text(min_size=1, max_size=10))
    num_nulls = draw(st.integers(min_value=1, max_value=3))
    return prefix + '\x00' * num_nulls


@given(text_ending_with_null(), st.integers(min_value=1, max_value=5))
@settings(max_examples=200)
def test_bug_multiply_strips_trailing_nulls(s, n):
    arr = np.array([s])
    result = char.multiply(arr, n)[0]
    expected = s * n
    assert result == expected, f"Failed: char.multiply({repr(s)}, {n}) = {repr(result)}, expected {repr(expected)}"


if __name__ == "__main__":
    test_bug_multiply_strips_trailing_nulls()
```

<details>

<summary>
**Failing input**: `s='0\x00', n=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 27, in <module>
    test_bug_multiply_strips_trailing_nulls()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 18, in test_bug_multiply_strips_trailing_nulls
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 23, in test_bug_multiply_strips_trailing_nulls
    assert result == expected, f"Failed: char.multiply({repr(s)}, {n}) = {repr(result)}, expected {repr(expected)}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Failed: char.multiply('0\x00', 1) = np.str_('0'), expected '0\x00'
Falsifying example: test_bug_multiply_strips_trailing_nulls(
    # The test always failed when commented parts were varied together.
    s='0\x00',  # or any other generated value
    n=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env')

import numpy as np
import numpy.char as char

# Test case 1: Basic example from the bug report
print("=== Test Case 1: Basic Example ===")
arr = np.array(['hello\x00'])
result = char.multiply(arr, 2)[0]
expected = 'hello\x00' * 2

print(f"Input: 'hello\\x00' * 2")
print(f"char.multiply result: {repr(result)}")
print(f"Expected: {repr(expected)}")
print(f"Match: {result == expected}")
print(f"Result length: {len(result)}")
print(f"Expected length: {len(expected)}")
print()

# Test case 2: Minimal failing case from hypothesis
print("=== Test Case 2: Minimal Case ===")
arr = np.array(['0\x00'])
result = char.multiply(arr, 1)[0]
expected = '0\x00' * 1

print(f"Input: '0\\x00' * 1")
print(f"char.multiply result: {repr(result)}")
print(f"Expected: {repr(expected)}")
print(f"Match: {result == expected}")
print(f"Result length: {len(result)}")
print(f"Expected length: {len(expected)}")
print()

# Test case 3: Multiple null bytes
print("=== Test Case 3: Multiple Null Bytes ===")
arr = np.array(['abc\x00\x00\x00'])
result = char.multiply(arr, 3)[0]
expected = 'abc\x00\x00\x00' * 3

print(f"Input: 'abc\\x00\\x00\\x00' * 3")
print(f"char.multiply result: {repr(result)}")
print(f"Expected: {repr(expected)}")
print(f"Match: {result == expected}")
print(f"Result length: {len(result)}")
print(f"Expected length: {len(expected)}")
print()

# Test case 4: Only null bytes
print("=== Test Case 4: Only Null Bytes ===")
arr = np.array(['\x00'])
result = char.multiply(arr, 5)[0]
expected = '\x00' * 5

print(f"Input: '\\x00' * 5")
print(f"char.multiply result: {repr(result)}")
print(f"Expected: {repr(expected)}")
print(f"Match: {result == expected}")
print(f"Result length: {len(result)}")
print(f"Expected length: {len(expected)}")
print()

# Test case 5: Python native string multiplication for comparison
print("=== Python Native String Multiplication ===")
test_strings = ['hello\x00', '0\x00', 'abc\x00\x00\x00', '\x00']
for s in test_strings:
    python_result = s * 3
    print(f"'{repr(s)}' * 3 = {repr(python_result)} (length: {len(python_result)})")
```

<details>

<summary>
All test cases demonstrate null byte stripping
</summary>
```
=== Test Case 1: Basic Example ===
Input: 'hello\x00' * 2
char.multiply result: np.str_('hellohello')
Expected: 'hello\x00hello\x00'
Match: False
Result length: 10
Expected length: 12

=== Test Case 2: Minimal Case ===
Input: '0\x00' * 1
char.multiply result: np.str_('0')
Expected: '0\x00'
Match: False
Result length: 1
Expected length: 2

=== Test Case 3: Multiple Null Bytes ===
Input: 'abc\x00\x00\x00' * 3
char.multiply result: np.str_('abcabcabc')
Expected: 'abc\x00\x00\x00abc\x00\x00\x00abc\x00\x00\x00'
Match: False
Result length: 9
Expected length: 18

=== Test Case 4: Only Null Bytes ===
Input: '\x00' * 5
char.multiply result: np.str_('')
Expected: '\x00\x00\x00\x00\x00'
Match: False
Result length: 0
Expected length: 5

=== Python Native String Multiplication ===
''hello\x00'' * 3 = 'hello\x00hello\x00hello\x00' (length: 18)
''0\x00'' * 3 = '0\x000\x000\x00' (length: 6)
''abc\x00\x00\x00'' * 3 = 'abc\x00\x00\x00abc\x00\x00\x00abc\x00\x00\x00' (length: 18)
''\x00'' * 3 = '\x00\x00\x00' (length: 3)
```
</details>

## Why This Is A Bug

This behavior violates the expected semantics of string multiplication for several critical reasons:

1. **Documentation Contradiction**: The numpy.char.multiply documentation states it returns "(a * i), that is string multiple concatenation, element-wise", explicitly suggesting equivalence with Python's native `*` operator for strings. However, Python preserves null bytes while numpy.char.multiply silently strips them.

2. **Silent Data Corruption**: The function removes trailing null bytes without any warning, error, or documentation of this limitation. This is particularly dangerous because:
   - Applications may rely on null bytes as delimiters or padding
   - Binary data encoded as strings will be corrupted
   - The data loss occurs silently, making it difficult to detect

3. **Inconsistent Behavior**: When multiplying a string containing null bytes by 1, the function should return an identical string, but instead returns a modified string with null bytes removed. This violates the mathematical identity property of multiplication by 1.

4. **No Documented Limitation**: Neither the numpy.char.multiply nor numpy.strings.multiply documentation mentions any special handling or limitations regarding null bytes.

## Relevant Context

The bug appears to stem from the underlying C implementation (`_multiply_ufunc`) which likely uses null-terminated string operations. The implementation is found in:
- `/numpy/_core/defchararray.py:279-328` - High-level wrapper
- `/numpy/_core/strings.py:166-226` - Main multiply implementation
- The actual multiplication is performed by `_multiply_ufunc` from numpy._core.umath

Similar issues may affect other string operations in the numpy.char module that rely on C string functions treating null bytes as terminators.

Documentation references:
- [NumPy char module documentation](https://numpy.org/doc/stable/reference/routines.char.html)
- The char module is marked as legacy, with numpy.strings as the recommended replacement, but the bug affects both modules

## Proposed Fix

The fix requires modifying the underlying C implementation to use length-aware string operations instead of null-terminated string functions. Since the exact C implementation is not directly accessible in the Python source, here's a high-level approach:

1. Track the actual string length including null bytes
2. Use memory operations (like memcpy) instead of string operations (like strcpy)
3. Ensure the output buffer size accounts for all characters including nulls
4. Preserve null bytes during the concatenation process

The Python-level code in strings.py correctly calculates buffer sizes based on str_len, but the underlying _multiply_ufunc appears to be truncating at null bytes. The fix would need to be applied at the C level where _multiply_ufunc is implemented, ensuring it respects the full string length rather than stopping at null terminators.