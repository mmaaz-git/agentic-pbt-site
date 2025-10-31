# Bug Report: numpy.strings.slice Incorrectly Handles Explicit None Stop Parameter

**Target**: `numpy.strings.slice`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.strings.slice` function returns incorrect results when `stop=None` is explicitly passed, either with or without a `step` parameter, violating Python's standard slicing semantics where `None` means "to the end".

## Property-Based Test

```python
import numpy as np
import numpy.strings as ns
from hypothesis import given, strategies as st, settings


@given(
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=20),
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=1000)
def test_slice_none_end_with_step(string_list, start, step):
    arr = np.array(string_list)
    result = ns.slice(arr, start, None, step)

    for i, s in enumerate(arr):
        expected = s[start:None:step]
        actual = result[i]
        assert actual == expected, f"Failed for string '{s}': expected '{expected}', got '{actual}'"

if __name__ == "__main__":
    test_slice_none_end_with_step()
```

<details>

<summary>
**Failing input**: `string_list=['0'], start=0, step=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 22, in <module>
    test_slice_none_end_with_step()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 7, in test_slice_none_end_with_step
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 19, in test_slice_none_end_with_step
    assert actual == expected, f"Failed for string '{s}': expected '{expected}', got '{actual}'"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Failed for string '0': expected '0', got ''
Falsifying example: test_slice_none_end_with_step(
    string_list=['0'],
    start=0,
    step=2,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as ns

# Test case 1: slice with explicit None stop and step
arr = np.array(["hello", "world"])
result1 = ns.slice(arr, 0, None, 2)
print("Test 1: ns.slice(['hello', 'world'], 0, None, 2)")
print("  Result:", result1)
print("  Expected: ['hlo', 'wrd'] (from 'hello'[0:None:2] = 'hlo')")

# Test case 2: minimal failing example from Hypothesis
arr2 = np.array(['0'])
result2 = ns.slice(arr2, 0, None, 2)
print("\nTest 2: ns.slice(['0'], 0, None, 2)")
print("  Result:", result2)
print("  Expected: ['0'] (from '0'[0:None:2] = '0')")

# Test case 3: slice with explicit None stop, no step
arr3 = np.array(["hello", "world"])
result3 = ns.slice(arr3, 2, None)
print("\nTest 3: ns.slice(['hello', 'world'], 2, None)")
print("  Result:", result3)
print("  Expected: ['llo', 'rld'] (from 'hello'[2:None] = 'llo')")

# Demonstrating correct Python behavior for comparison
print("\n--- Python's built-in slicing behavior ---")
print("'hello'[0:None:2] =", 'hello'[0:None:2])
print("'world'[0:None:2] =", 'world'[0:None:2])
print("'0'[0:None:2] =", '0'[0:None:2])
print("'hello'[2:None] =", 'hello'[2:None])
print("'world'[2:None] =", 'world'[2:None])
```

<details>

<summary>
Output showing incorrect results from numpy.strings.slice
</summary>
```
Test 1: ns.slice(['hello', 'world'], 0, None, 2)
  Result: ['' '']
  Expected: ['hlo', 'wrd'] (from 'hello'[0:None:2] = 'hlo')

Test 2: ns.slice(['0'], 0, None, 2)
  Result: ['']
  Expected: ['0'] (from '0'[0:None:2] = '0')

Test 3: ns.slice(['hello', 'world'], 2, None)
  Result: ['he' 'wo']
  Expected: ['llo', 'rld'] (from 'hello'[2:None] = 'llo')

--- Python's built-in slicing behavior ---
'hello'[0:None:2] = hlo
'world'[0:None:2] = wrd
'0'[0:None:2] = 0
'hello'[2:None] = llo
'world'[2:None] = rld
```
</details>

## Why This Is A Bug

This bug violates the fundamental contract that `numpy.strings.slice` should behave "like in the regular Python `slice` object" as stated in its documentation.

In Python's standard slicing semantics, `None` has a specific meaning:
- When used as `stop`, it means "slice to the end of the string"
- `"hello"[0:None:2]` correctly returns `"hlo"` (every 2nd character from start to end)
- `"hello"[2:None]` correctly returns `"llo"` (from index 2 to end)

However, `numpy.strings.slice` incorrectly handles explicit `None` values due to flawed swap logic at lines 1804-1806 in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/strings.py`:

```python
if stop is None:
    stop = start
    start = None
```

This swap is intended to handle the single-argument case where `slice(3)` should behave like `slice(None, 3)` (slice up to position 3). However, it incorrectly triggers when `stop=None` is explicitly passed with other arguments:

1. **With step parameter**: `slice(0, None, 2)` becomes `slice(None, 0, 2)` after the swap, which then gets interpreted as `slice(0, 0, 2)` - an empty slice.

2. **Without step parameter**: `slice(2, None)` becomes `slice(None, 2)` after the swap, which slices from beginning to position 2 instead of from position 2 to end.

This causes silent data corruption where the function returns wrong slices without any error or warning.

## Relevant Context

The bug affects NumPy version 2.3.0 and likely earlier versions that contain the same swap logic. The issue manifests in two distinct scenarios:

1. **Slicing with explicit None and step**: Returns empty strings instead of the correct slice
2. **Slicing with explicit None without step**: Returns the inverse slice (beginning to position instead of position to end)

The documentation at line 1749-1750 states: "Like in the regular Python `slice` object, if only `start` is specified then it is interpreted as the `stop`." This correctly describes the single-argument behavior but the implementation incorrectly applies this swap even when multiple arguments are provided.

Python's built-in `slice` object documentation: https://docs.python.org/3/library/functions.html#slice
NumPy strings module source: https://github.com/numpy/numpy/blob/main/numpy/_core/strings.py

## Proposed Fix

The swap logic should only apply when the function is called with a single positional argument. When `step` is explicitly provided or when both `start` and `stop` are provided (even if `stop` is `None`), the swap should not occur:

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -1801,7 +1801,7 @@ def slice(a, start=None, stop=None, step=None, /):
     """
     # Just like in the construction of a regular slice object, if only start
     # is specified then start will become stop, see logic in slice_new.
-    if stop is None:
+    if stop is None and step is None:
         stop = start
         start = None
```