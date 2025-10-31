# Bug Report: numpy.strings.find Always Returns 0 for Null Character Searches

**Target**: `numpy.strings.find`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.strings.find()` function incorrectly returns 0 (or the start position) instead of -1 when searching for null characters (`'\x00'`) that are not present in the string, and returns 0 even when null characters ARE present at other positions.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings, example

@given(st.lists(st.text(min_size=1), min_size=1).map(lambda x: np.array(x, dtype=str)),
       st.text(min_size=1, max_size=5),
       st.integers(min_value=0, max_value=20),
       st.one_of(st.integers(min_value=0, max_value=20), st.none()))
@example(arr=np.array(['abc'], dtype=str), sub='\x00', start=0, end=None)
@settings(max_examples=1000)
def test_find_with_bounds(arr, sub, start, end):
    result = nps.find(arr, sub, start, end)
    for i in range(len(arr)):
        expected = arr[i].find(sub, start, end)
        assert result[i] == expected, f"Failed for arr[{i}]='{arr[i]}', sub='{sub}', start={start}, end={end}: expected {expected}, got {result[i]}"

# Run the test
if __name__ == "__main__":
    test_find_with_bounds()
```

<details>

<summary>
**Failing input**: `arr=np.array(['abc'], dtype=str), sub='\x00', start=0, end=None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 19, in <module>
    test_find_with_bounds()
    ~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 6, in test_find_with_bounds
    st.text(min_size=1, max_size=5),
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 15, in test_find_with_bounds
    assert result[i] == expected, f"Failed for arr[{i}]='{arr[i]}', sub='{sub}', start={start}, end={end}: expected {expected}, got {result[i]}"
           ^^^^^^^^^^^^^^^^^^^^^
AssertionError: Failed for arr[0]='abc', sub=' ', start=0, end=None: expected -1, got 0
Falsifying explicit example: test_find_with_bounds(
    arr=array(['abc'], dtype='<U3'),
    sub='\x00',
    start=0,
    end=None,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

print("Testing numpy.strings.find with null character '\\x00':")
print("=" * 60)

test_cases = [
    '',
    'abc',
    'a\x00b',
    '\x00',
    'hello\x00world',
    'test'
]

print(f"{'String':<20} {'Python find':<15} {'NumPy find':<15} {'Match?':<10}")
print("-" * 60)

for s in test_cases:
    arr = np.array([s], dtype=str)
    np_find = nps.find(arr, '\x00')[0]
    py_find = s.find('\x00')
    match = "✓" if np_find == py_find else "✗"

    # Format string representation
    s_repr = repr(s) if len(repr(s)) < 20 else repr(s)[:17] + "..."

    print(f"{s_repr:<20} {py_find:<15} {np_find:<15} {match:<10}")

print("\nAdditional test with start parameter:")
print("-" * 60)
s = 'abcdefg'
arr = np.array([s], dtype=str)
start = 2

np_find_with_start = nps.find(arr, '\x00', start)[0]
py_find_with_start = s.find('\x00', start)

print(f"find('{s}', '\\x00', start={start}):")
print(f"  Python: {py_find_with_start}")
print(f"  NumPy:  {np_find_with_start}")
print(f"  Match: {'✓' if np_find_with_start == py_find_with_start else '✗'}")
```

<details>

<summary>
Output showing incorrect behavior for null character searches
</summary>
```
Testing numpy.strings.find with null character '\x00':
============================================================
String               Python find     NumPy find      Match?
------------------------------------------------------------
''                   -1              0               ✗
'abc'                -1              0               ✗
'a\x00b'             1               0               ✗
'\x00'               0               0               ✓
'hello\x00world'     5               0               ✗
'test'               -1              0               ✗

Additional test with start parameter:
------------------------------------------------------------
find('abcdefg', '\x00', start=2):
  Python: -1
  NumPy:  2
  Match: ✗
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Violates documented API contract**: The numpy.strings.find documentation explicitly states "See Also: str.find", indicating it should mimic Python's str.find() behavior. Python's str.find() returns -1 when a substring is not found, including for null characters.

2. **Returns wrong position when found**: When the null character IS present (e.g., 'a\x00b'), Python correctly returns position 1, but NumPy incorrectly returns 0. Similarly, 'hello\x00world' has the null at position 5, but NumPy returns 0.

3. **Start parameter handling is broken**: When using a start parameter, NumPy returns the start position itself instead of -1 when the null character is not found. For example, find('abcdefg', '\x00', start=2) returns 2 instead of -1.

4. **Makes null character searches impossible**: Users cannot reliably determine if a null character exists in a string or where it's located. The function always returns 0 (or the start position) regardless of the actual presence or location of null characters.

5. **Silent data corruption risk**: Applications processing binary data, network protocols, or file formats with embedded nulls will get incorrect results without any error or warning, potentially leading to data corruption or security issues.

## Relevant Context

The issue appears to be in NumPy's C implementation of the find ufunc (universal function). The function is treating null characters as zero-width patterns that always match at the search start position, rather than as regular single-byte characters that should be searched for normally.

This bug affects NumPy version 2.3.0 and likely earlier versions. The find function is imported from `numpy._core.umath` as `_find_ufunc` and is implemented in NumPy's C extension code.

Documentation reference: https://numpy.org/doc/stable/reference/generated/numpy.strings.find.html

## Proposed Fix

Since the bug is in NumPy's C implementation (the `_find_ufunc` in `numpy._core.umath`), a fix would require modifying the C code that implements the string find operation. The fix should:

1. Remove any special handling for null characters in the find implementation
2. Treat '\x00' as a regular searchable character like any other byte
3. Ensure the function returns -1 when null characters are not found
4. Return the correct position when null characters are found

Without access to the C source code, I cannot provide a specific patch, but the implementation should follow Python's str.find() semantics exactly, including for null character searches.