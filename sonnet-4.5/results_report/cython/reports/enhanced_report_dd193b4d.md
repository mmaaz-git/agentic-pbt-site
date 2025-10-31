# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Values for Negative Step Slices

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function returns negative values when given slices with negative steps, violating the fundamental principle that lengths must be non-negative integers.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    n=st.integers(min_value=1, max_value=1000),
    start=st.integers(min_value=-100, max_value=100) | st.none(),
    stop=st.integers(min_value=-100, max_value=100) | st.none(),
    step=st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0) | st.none(),
)
@settings(max_examples=500)
def test_length_of_indexer_slice_matches_actual_length(n, start, stop, step):
    target = np.arange(n)
    indexer = slice(start, stop, step)

    expected_len = length_of_indexer(indexer, target)
    actual_len = len(target[indexer])

    assert expected_len == actual_len, f"For n={n}, slice({start}, {stop}, {step}): length_of_indexer returned {expected_len}, but actual length is {actual_len}"

if __name__ == "__main__":
    test_length_of_indexer_slice_matches_actual_length()
```

<details>

<summary>
**Failing input**: `n=1, start=None, stop=None, step=-1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 22, in <module>
    test_length_of_indexer_slice_matches_actual_length()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 6, in test_length_of_indexer_slice_matches_actual_length
    n=st.integers(min_value=1, max_value=1000),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 19, in test_length_of_indexer_slice_matches_actual_length
    assert expected_len == actual_len, f"For n={n}, slice({start}, {stop}, {step}): length_of_indexer returned {expected_len}, but actual length is {actual_len}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: For n=1, slice(None, None, -1): length_of_indexer returned -1, but actual length is 1
Falsifying example: test_length_of_indexer_slice_matches_actual_length(
    n=1,
    start=None,
    stop=None,
    step=-1,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.indexers import length_of_indexer

# Test case from the bug report
target = np.array([0])
slc = slice(None, None, -1)

result = length_of_indexer(slc, target)
actual = len(target[slc])

print(f"Test 1: Array with single element [0], slice(None, None, -1)")
print(f"  length_of_indexer returned: {result}")
print(f"  Actual length: {actual}")
print(f"  Match: {result == actual}")
print()

# Additional test cases from the report
test_cases = [
    (np.arange(5), slice(None, None, -1), "Array [0,1,2,3,4], slice(None, None, -1)"),
    (np.arange(10), slice(None, None, -2), "Array [0..9], slice(None, None, -2)"),
    (np.arange(10), slice(5, None, -1), "Array [0..9], slice(5, None, -1)"),
]

for i, (arr, slc, desc) in enumerate(test_cases, 2):
    result = length_of_indexer(slc, arr)
    actual = len(arr[slc])
    print(f"Test {i}: {desc}")
    print(f"  length_of_indexer returned: {result}")
    print(f"  Actual length: {actual}")
    print(f"  Match: {result == actual}")
    print()

# The assertion that fails
print("Running assertion from bug report:")
try:
    target = np.array([0])
    slc = slice(None, None, -1)
    result = length_of_indexer(slc, target)
    actual = len(target[slc])
    assert result == actual, f"Expected {actual}, got {result}"
    print("Assertion passed")
except AssertionError as e:
    print(f"AssertionError: {e}")
```

<details>

<summary>
Output: Function returns negative values for all negative-step slices
</summary>
```
Test 1: Array with single element [0], slice(None, None, -1)
  length_of_indexer returned: -1
  Actual length: 1
  Match: False

Test 2: Array [0,1,2,3,4], slice(None, None, -1)
  length_of_indexer returned: -5
  Actual length: 5
  Match: False

Test 3: Array [0..9], slice(None, None, -2)
  length_of_indexer returned: -5
  Actual length: 5
  Match: False

Test 4: Array [0..9], slice(5, None, -1)
  length_of_indexer returned: -5
  Actual length: 6
  Match: False

Running assertion from bug report:
AssertionError: Expected 1, got -1
```
</details>

## Why This Is A Bug

This violates the fundamental contract of a length function in several ways:

1. **Semantic violation**: The function name `length_of_indexer` and its docstring "Return the expected length of target[indexer]" clearly indicate it should return a length. By definition in programming, lengths are non-negative integers representing counts of elements.

2. **Python convention violation**: Python's built-in `len()` function always returns non-negative integers. For any valid array `a` and slice `s`, `len(a[s])` is always >= 0. The function claims to return "the expected length of target[indexer]" but returns values that contradict `len(target[indexer])`.

3. **Mathematical incorrectness**: The function returns the negation of the actual length for negative-step slices. For example, an array of 5 elements sliced with `[::-1]` has 5 elements, not -5.

4. **Breaks downstream code**: Any code using this function to allocate buffers, iterate, or perform calculations based on expected array lengths will fail catastrophically when receiving negative values.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexers/utils.py` at lines 303-316. The function attempts to handle negative steps by swapping start/stop and negating the step (lines 313-315), but fails to properly set default values for `None` start/stop values when the step is negative.

When `step < 0`:
- A `None` start should default to the last valid index (`target_len - 1`), not 0
- A `None` stop should default to before the first element (conceptually `-target_len - 1`), not `target_len`

The current code incorrectly applies positive-step defaults (lines 303-310) before checking the step sign, causing the calculation to produce negative results.

Documentation: https://pandas.pydata.org/docs/reference/api/pandas.core.indexers.length_of_indexer.html

## Proposed Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -300,17 +300,25 @@ def length_of_indexer(indexer, target=None) -> int:
         start = indexer.start
         stop = indexer.stop
         step = indexer.step
+
+        if step is None:
+            step = 1
+
         if start is None:
-            start = 0
+            if step < 0:
+                start = target_len - 1
+            else:
+                start = 0
         elif start < 0:
             start += target_len
-        if stop is None or stop > target_len:
-            stop = target_len
+
+        if stop is None:
+            if step < 0:
+                stop = -target_len - 1
+            else:
+                stop = target_len
+        elif stop > target_len:
+            stop = target_len
         elif stop < 0:
             stop += target_len
-        if step is None:
-            step = 1
-        elif step < 0:
+
+        if step < 0:
             start, stop = stop + 1, start + 1
             step = -step
         return (stop - start + step - 1) // step
```