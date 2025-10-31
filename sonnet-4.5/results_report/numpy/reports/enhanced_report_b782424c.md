# Bug Report: numpy.strings.index Raises ValueError for Arrays with Mixed Results

**Target**: `numpy.strings.index` and `numpy.strings.rindex`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.strings.index` and `numpy.strings.rindex` raise `ValueError` when ANY element in the input array doesn't contain the substring, breaking vectorized operations and making these functions unusable for real-world data processing with mixed results.

## Property-Based Test

```python
import numpy as np
import numpy.strings as ns
from hypothesis import given, settings, strategies as st

string_arrays_with_substring = st.lists(st.text(), min_size=1, max_size=10).flatmap(
    lambda strings: st.tuples(st.just(np.array(strings)), st.sampled_from(strings))
)

@given(string_arrays_with_substring)
@settings(max_examples=500)
def test_find_index_consistency(arr_and_sub):
    arr, sub = arr_and_sub
    find_result = ns.find(arr, sub)

    for i in range(len(arr)):
        if find_result[i] >= 0:
            try:
                index_result = ns.index(arr, sub)
                assert find_result[i] == index_result[i]
            except ValueError:
                assert False, f"find returned {find_result[i]} but index raised ValueError"

if __name__ == "__main__":
    test_find_index_consistency()
```

<details>

<summary>
**Failing input**: `arr_and_sub=(array(['', '', '', '0'], dtype='<U1'), '0')`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 18, in test_find_index_consistency
    index_result = ns.index(arr, sub)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/strings.py", line 383, in index
    return _index_ufunc(a, sub, start, end)
ValueError: substring not found

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 24, in <module>
    test_find_index_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 10, in test_find_index_consistency
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 21, in test_find_index_consistency
    assert False, f"find returned {find_result[i]} but index raised ValueError"
           ^^^^^
AssertionError: find returned 0 but index raised ValueError
Falsifying example: test_find_index_consistency(
    arr_and_sub=(array(['', '', '', '0'], dtype='<U1'), '0'),
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as ns

# Test case that demonstrates the bug
arr = np.array(['0', ''])
sub = '0'

print("="*60)
print("Testing numpy.strings.index with mixed results array")
print("="*60)
print(f"Input array: {repr(arr)}")
print(f"Substring to find: {repr(sub)}")
print()

# Test find - this should work correctly
print("1. Testing numpy.strings.find (expected to work):")
find_result = ns.find(arr, sub)
print(f"   find(arr, '0') returns: {find_result}")
print(f"   Element 0 ('0'): substring found at index {find_result[0]}")
print(f"   Element 1 (''): substring not found (returns {find_result[1]})")
print()

# Test index - this should raise ValueError
print("2. Testing numpy.strings.index (demonstrates bug):")
try:
    index_result = ns.index(arr, sub)
    print(f"   index(arr, '0') returns: {index_result}")
except ValueError as e:
    print(f"   index(arr, '0') raises ValueError: {e}")
    print()
    print("   BUG CONFIRMED: find() correctly returns [0, -1] for mixed results,")
    print("   but index() raises ValueError for the entire array operation")
    print("   when ANY element doesn't contain the substring.")

print()
print("="*60)
print("Testing numpy.strings.rindex with same array")
print("="*60)

# Test rfind - this should work correctly
print("3. Testing numpy.strings.rfind (expected to work):")
rfind_result = ns.rfind(arr, sub)
print(f"   rfind(arr, '0') returns: {rfind_result}")
print()

# Test rindex - this should also raise ValueError
print("4. Testing numpy.strings.rindex (demonstrates bug):")
try:
    rindex_result = ns.rindex(arr, sub)
    print(f"   rindex(arr, '0') returns: {rindex_result}")
except ValueError as e:
    print(f"   rindex(arr, '0') raises ValueError: {e}")
    print()
    print("   BUG CONFIRMED: rfind() correctly returns [0, -1] for mixed results,")
    print("   but rindex() raises ValueError for the entire array operation")
    print("   when ANY element doesn't contain the substring.")

print()
print("="*60)
print("Additional test case with more elements")
print("="*60)

# Test with more complex array
arr2 = np.array(['hello world', 'goodbye', 'world peace'])
sub2 = 'world'

print(f"Input array: {repr(arr2)}")
print(f"Substring to find: {repr(sub2)}")
print()

print("5. Testing find with larger array:")
find_result2 = ns.find(arr2, sub2)
print(f"   find(arr2, 'world') returns: {find_result2}")
print(f"   - 'hello world': found at index {find_result2[0]}")
print(f"   - 'goodbye': not found (returns {find_result2[1]})")
print(f"   - 'world peace': found at index {find_result2[2]}")
print()

print("6. Testing index with larger array:")
try:
    index_result2 = ns.index(arr2, sub2)
    print(f"   index(arr2, 'world') returns: {index_result2}")
except ValueError as e:
    print(f"   index(arr2, 'world') raises ValueError: {e}")
    print()
    print("   The function fails for ANY array with mixed results,")
    print("   making it unusable for real-world vectorized operations.")
```

<details>

<summary>
ValueError raised for array with mixed substring presence
</summary>
```
============================================================
Testing numpy.strings.index with mixed results array
============================================================
Input array: array(['0', ''], dtype='<U1')
Substring to find: '0'

1. Testing numpy.strings.find (expected to work):
   find(arr, '0') returns: [ 0 -1]
   Element 0 ('0'): substring found at index 0
   Element 1 (''): substring not found (returns -1)

2. Testing numpy.strings.index (demonstrates bug):
   index(arr, '0') raises ValueError: substring not found

   BUG CONFIRMED: find() correctly returns [0, -1] for mixed results,
   but index() raises ValueError for the entire array operation
   when ANY element doesn't contain the substring.

============================================================
Testing numpy.strings.rindex with same array
============================================================
3. Testing numpy.strings.rfind (expected to work):
   rfind(arr, '0') returns: [ 0 -1]

4. Testing numpy.strings.rindex (demonstrates bug):
   rindex(arr, '0') raises ValueError: substring not found

   BUG CONFIRMED: rfind() correctly returns [0, -1] for mixed results,
   but rindex() raises ValueError for the entire array operation
   when ANY element doesn't contain the substring.

============================================================
Additional test case with more elements
============================================================
Input array: array(['hello world', 'goodbye', 'world peace'], dtype='<U11')
Substring to find: 'world'

5. Testing find with larger array:
   find(arr2, 'world') returns: [ 6 -1  0]
   - 'hello world': found at index 6
   - 'goodbye': not found (returns -1)
   - 'world peace': found at index 0

6. Testing index with larger array:
   index(arr2, 'world') raises ValueError: substring not found

   The function fails for ANY array with mixed results,
   making it unusable for real-world vectorized operations.
```
</details>

## Why This Is A Bug

This violates NumPy's fundamental design principle of element-wise operations on arrays. The documentation states that `index` is "Like `find`, but raises `ValueError` when the substring is not found" and promises to return "ndarray - Output array of ints". However:

1. **Contract violation**: The function's return type is documented as "ndarray" but it raises an exception instead when processing arrays with mixed results (where some elements contain the substring and others don't).

2. **Inconsistent with find/rfind**: The `find` and `rfind` functions correctly return arrays with -1 for elements where the substring is not found, allowing proper vectorized processing. The `index` and `rindex` functions fail completely on the same input.

3. **Breaks vectorization**: The current behavior makes these functions unusable for real-world data processing where not all strings in an array are guaranteed to contain a given substring. This is a common scenario in data analysis.

4. **Misleading documentation**: The phrase "Like `find`" implies similar vectorized behavior with different error handling for individual elements, not a complete failure of the entire array operation.

5. **Unexpected behavior**: Users expect NumPy functions to operate element-wise. Having the entire operation fail because one element doesn't contain the substring violates this core expectation.

## Relevant Context

The issue stems from the underlying C implementation of the `_index_ufunc` and `_rindex_ufunc` functions imported from `numpy._core.umath`. These ufuncs raise a ValueError at the C level when any element doesn't contain the substring, rather than handling the "not found" case element-wise.

The functions are located in `/numpy/_core/strings.py`:
- `index` function at line 353-383
- `rindex` function at line 387-417

Both functions are thin wrappers around their respective ufuncs:
```python
def index(a, sub, start=0, end=None):
    end = end if end is not None else MAX
    return _index_ufunc(a, sub, start, end)

def rindex(a, sub, start=0, end=None):
    end = end if end is not None else MAX
    return _rindex_ufunc(a, sub, start, end)
```

Example from the `rfind` documentation (line 343-345) shows the expected behavior for arrays with mixed results:
```python
>>> b = np.array(["Computer Science", "Science"])
>>> np.strings.rfind(b, "Science", start=0, end=None)
array([9, 0])
```

This demonstrates that NumPy string functions are designed to handle arrays where the substring appears at different positions (or not at all) in different elements.

## Proposed Fix

The proper fix requires modifying the underlying C/C++ ufuncs to handle the "not found" case element-wise. However, a Python-level workaround could validate the operation before raising:

```diff
@set_module("numpy.strings")
def index(a, sub, start=0, end=None):
    """
    Like `find`, but raises :exc:`ValueError` when the substring is not found.

    ...
    """
    end = end if end is not None else MAX
-   return _index_ufunc(a, sub, start, end)
+   # First use find to get the results
+   result = _find_ufunc(a, sub, start, end)
+   # Check if any element didn't find the substring
+   if np.any(result < 0):
+       # For consistency with Python's str.index, raise ValueError
+       # but this should ideally be element-wise, not array-wise
+       raise ValueError("substring not found")
+   return result

@set_module("numpy.strings")
def rindex(a, sub, start=0, end=None):
    """
    Like `rfind`, but raises :exc:`ValueError` when the substring `sub` is
    not found.

    ...
    """
    end = end if end is not None else MAX
-   return _rindex_ufunc(a, sub, start, end)
+   # First use rfind to get the results
+   result = _rfind_ufunc(a, sub, start, end)
+   # Check if any element didn't find the substring
+   if np.any(result < 0):
+       # For consistency with Python's str.rindex, raise ValueError
+       # but this should ideally be element-wise, not array-wise
+       raise ValueError("substring not found")
+   return result
```

However, this is still not ideal as it raises for the entire array. A better solution would be to modify the ufuncs at the C level to return valid results for all elements and potentially add a separate method for checking if all searches succeeded, or to return a masked array where failed searches are masked.