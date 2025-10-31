# Bug Report: numpy.strings.slice() Incorrectly Handles Explicit None as Stop Parameter

**Target**: `numpy.strings.slice()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `stop=None` is explicitly passed to `numpy.strings.slice()`, the function incorrectly swaps parameters and treats `start` as `stop`, returning only the first `start` characters instead of slicing from `start` to the end of the string as expected.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, example

@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10))
@example(['hello'])
def test_slice_with_none_stop(str_list):
    """Property: nps.slice(arr, start, None) should behave like Python arr[start:]"""
    arr = np.array(str_list, dtype='U')
    result = nps.slice(arr, 0, None)

    for i in range(len(arr)):
        expected = str_list[i][0:]
        assert result[i] == expected, f"slice(arr, 0, None) failed: got '{result[i]}', expected '{expected}'"
```

<details>

<summary>
**Failing input**: `['hello']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 17, in <module>
    test_slice_with_none_stop()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 6, in test_slice_with_none_stop
    @example(['hello'])
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 14, in test_slice_with_none_stop
    assert result[i] == expected, f"slice(arr, 0, None) failed: got '{result[i]}', expected '{expected}'"
           ^^^^^^^^^^^^^^^^^^^^^
AssertionError: slice(arr, 0, None) failed: got '', expected 'hello'
Falsifying explicit example: test_slice_with_none_stop(
    str_list=['hello'],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

# Test case 1: slice(arr, 1, None) should slice from index 1 to end
arr = np.array(['hello', 'world', 'test'])

result = nps.slice(arr, 1, None)
print(f"Test 1: nps.slice(['hello', 'world', 'test'], 1, None)")
print(f"Result:   {result}")
print(f"Expected: {np.array(['ello', 'orld', 'est'])}")
print()

# Test case 2: slice(arr, 0, None) should return full strings
arr2 = np.array(['hello'])
result2 = nps.slice(arr2, 0, None)
print(f"Test 2: nps.slice(['hello'], 0, None)")
print(f"Result:   {result2}")
print(f"Expected: {np.array(['hello'])}")
print()

# Test case 3: slice(arr, 2, None) should slice from index 2 to end
arr3 = np.array(['abcdef', 'ghijkl'])
result3 = nps.slice(arr3, 2, None)
print(f"Test 3: nps.slice(['abcdef', 'ghijkl'], 2, None)")
print(f"Result:   {result3}")
print(f"Expected: {np.array(['cdef', 'ijkl'])}")
print()

# Comparison with Python slicing behavior
print("Comparison with standard Python slicing:")
s = 'hello'
print(f"Python 'hello'[1:None] = '{s[1:None]}'")
print(f"Python 'hello'[1:]     = '{s[1:]}'")
print(f"numpy.strings.slice(['hello'], 1, None) = {nps.slice(np.array(['hello']), 1, None)}")
```

<details>

<summary>
Output demonstrating the incorrect behavior
</summary>
```
Test 1: nps.slice(['hello', 'world', 'test'], 1, None)
Result:   ['h' 'w' 't']
Expected: ['ello' 'orld' 'est']

Test 2: nps.slice(['hello'], 0, None)
Result:   ['']
Expected: ['hello']

Test 3: nps.slice(['abcdef', 'ghijkl'], 2, None)
Result:   ['ab' 'gh']
Expected: ['cdef' 'ijkl']

Comparison with standard Python slicing:
Python 'hello'[1:None] = 'ello'
Python 'hello'[1:]     = 'ello'
numpy.strings.slice(['hello'], 1, None) = ['h']
```
</details>

## Why This Is A Bug

The `numpy.strings.slice()` function is documented to work "Like in the regular Python `slice` object", where `s[start:None]` is equivalent to `s[start:]` (slice from start to end). However, when `stop=None` is explicitly passed as an argument, the function incorrectly invokes a special case that should only apply when stop is not provided at all.

The bug occurs because the implementation cannot distinguish between:
1. `slice(arr, 1)` - only one argument provided, should be treated as stop (special case)
2. `slice(arr, 1, None)` - two arguments provided with stop=None, should slice from 1 to end

The current implementation checks `if stop is None` which catches both cases, when it should only apply the special case when stop was not provided as an argument. This violates:

1. **Python slicing semantics**: In Python, `slice(1)` and `slice(1, None)` create different slice objects with different behaviors
2. **Documented behavior**: The function claims to behave "Like in the regular Python `slice` object"
3. **Principle of least surprise**: `None` universally means "no limit" in Python slicing, not "parameter not provided"
4. **Functional correctness**: The function produces incorrect results that don't match user expectations

## Relevant Context

The bug is in the source code of `numpy.strings.slice()` (found via `inspect.getsource()`):

```python
def slice(a, start=None, stop=None, step=None, /):
    # Just like in the construction of a regular slice object, if only start
    # is specified then start will become stop, see logic in slice_new.
    if stop is None:  # <- BUG: This catches both unspecified AND explicit None
        stop = start
        start = None
```

The comment references Python's `slice_new` logic for handling the special case where only one argument is provided. However, Python's built-in `slice()` correctly distinguishes between:
- `slice(5)` → slice(None, 5, None)  # stop=5
- `slice(5, None)` → slice(5, None, None)  # start=5, stop=None

This distinction is lost in the numpy implementation due to using None as both the default value and a legitimate parameter value.

## Proposed Fix

```diff
@set_module("numpy.strings")
-def slice(a, start=None, stop=None, step=None, /):
+_UNSPECIFIED = object()
+def slice(a, start=_UNSPECIFIED, stop=_UNSPECIFIED, step=_UNSPECIFIED, /):
     """
     Slice the strings in `a` by slices specified by `start`, `stop`, `step`.
     Like in the regular Python `slice` object, if only `start` is
     specified then it is interpreted as the `stop`.
     [... docstring continues ...]
     """
     # Just like in the construction of a regular slice object, if only start
     # is specified then start will become stop, see logic in slice_new.
-    if stop is None:
+    if stop is _UNSPECIFIED:
         stop = start
         start = None
+    else:
+        # Handle cases where parameters were explicitly set to None
+        if start is _UNSPECIFIED:
+            start = None
+        if stop is None:
+            stop = None  # Keep as None, will be handled below

     # adjust start, stop, step to be integers, see logic in PySlice_Unpack
-    if step is None:
+    if step is _UNSPECIFIED or step is None:
         step = 1
     step = np.asanyarray(step)
     if not np.issubdtype(step.dtype, np.integer):
         raise TypeError(f"unsupported type {step.dtype} for operand 'step'")
     if np.any(step == 0):
         raise ValueError("slice step cannot be zero")

     if start is None:
         start = np.where(step < 0, np.iinfo(np.intp).max, 0)

     if stop is None:
         stop = np.where(step < 0, np.iinfo(np.intp).min, np.iinfo(np.intp).max)

     return _slice(a, start, stop, step)
```