# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Length Values

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer()` function returns negative values for certain slice configurations, violating the mathematical property that lengths must be non-negative and contradicting its documented behavior of returning "the expected length of target[indexer]".

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.indexers import length_of_indexer


@settings(max_examples=1000)
@given(st.data())
def test_length_of_indexer_slice_oracle(data):
    target_len = data.draw(st.integers(min_value=0, max_value=1000))
    target = np.arange(target_len)

    start = data.draw(st.one_of(
        st.none(),
        st.integers(min_value=-target_len*2, max_value=target_len*2)
    ))
    stop = data.draw(st.one_of(
        st.none(),
        st.integers(min_value=-target_len*2, max_value=target_len*2)
    ))
    step = data.draw(st.one_of(
        st.none(),
        st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0)
    ))

    indexer = slice(start, stop, step)

    computed_length = length_of_indexer(indexer, target)
    actual_length = len(target[indexer])

    assert computed_length == actual_length, \
        f"Mismatch for slice({start}, {stop}, {step}) on array of length {target_len}: " \
        f"computed={computed_length}, actual={actual_length}"


if __name__ == "__main__":
    test_length_of_indexer_slice_oracle()
```

<details>

<summary>
**Failing input**: `slice(None, None, -1)` on array of length 1
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 39, in <module>
    test_length_of_indexer_slice_oracle()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 10, in test_length_of_indexer_slice_oracle
    @given(st.data())
                   ^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 33, in test_length_of_indexer_slice_oracle
    assert computed_length == actual_length, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Mismatch for slice(None, None, -1) on array of length 1: computed=-1, actual=1
Falsifying example: test_length_of_indexer_slice_oracle(
    data=data(...),
)
Draw 1: 1
Draw 2: None
Draw 3: None
Draw 4: -1
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/42/hypo.py:34
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from pandas.core.indexers import length_of_indexer

# Test the specific failing case from the bug report
target = np.array([0])
indexer = slice(1, 0, None)

computed = length_of_indexer(indexer, target)
actual = len(target[indexer])

print(f"Test case: slice(1, 0, None) on array [0]")
print(f"Computed length: {computed}")
print(f"Actual length: {actual}")
print(f"Bug confirmed: {computed} != {actual}")
print()

# Test additional cases to understand the scope
test_cases = [
    (np.arange(5), slice(None, None, -1), "Reverse slice on [0,1,2,3,4]"),
    (np.arange(3), slice(10, 20, None), "Out of bounds slice on [0,1,2]"),
    (np.arange(4), slice(-1, 0, None), "Negative to positive empty slice on [0,1,2,3]"),
    (np.array([42]), slice(None, None, -1), "Reverse slice on single element [42]"),
    (np.arange(10), slice(5, 3, None), "Empty slice(5, 3, None) on [0..9]"),
]

print("Additional test cases:")
print("-" * 50)
for target, indexer, description in test_cases:
    computed = length_of_indexer(indexer, target)
    actual = len(target[indexer])
    match = "✓" if computed == actual else "✗"
    print(f"{description}:")
    print(f"  Indexer: {indexer}")
    print(f"  Computed: {computed}, Actual: {actual} {match}")
    if computed != actual:
        print(f"  ERROR: Expected {actual}, got {computed}")
    print()
```

<details>

<summary>
Multiple failing cases demonstrating negative length returns
</summary>
```
Test case: slice(1, 0, None) on array [0]
Computed length: -1
Actual length: 0
Bug confirmed: -1 != 0

Additional test cases:
--------------------------------------------------
Reverse slice on [0,1,2,3,4]:
  Indexer: slice(None, None, -1)
  Computed: -5, Actual: 5 ✗
  ERROR: Expected 5, got -5

Out of bounds slice on [0,1,2]:
  Indexer: slice(10, 20, None)
  Computed: -7, Actual: 0 ✗
  ERROR: Expected 0, got -7

Negative to positive empty slice on [0,1,2,3]:
  Indexer: slice(-1, 0, None)
  Computed: -3, Actual: 0 ✗
  ERROR: Expected 0, got -3

Reverse slice on single element [42]:
  Indexer: slice(None, None, -1)
  Computed: -1, Actual: 1 ✗
  ERROR: Expected 1, got -1

Empty slice(5, 3, None) on [0..9]:
  Indexer: slice(5, 3, None)
  Computed: -2, Actual: 0 ✗
  ERROR: Expected 0, got -2

```
</details>

## Why This Is A Bug

The function violates its documented contract and fundamental mathematical principles:

1. **Documentation violation**: The docstring states the function should "Return the expected length of target[indexer]". When you perform `len(target[indexer])` in NumPy, you always get non-negative values, but `length_of_indexer` returns negative values.

2. **Mathematical impossibility**: Length is a non-negative measure by definition. In Python, `len()` always returns values ≥ 0. No collection can have a negative number of elements.

3. **Specific failing patterns identified**:
   - **Empty slices** where `start > stop` with positive step (e.g., `slice(1, 0, None)`) return negative values instead of 0
   - **Reverse slices** with negative step (e.g., `slice(None, None, -1)`) return negative values instead of the actual positive length
   - **Out-of-bounds slices** (e.g., `slice(10, 20, None)` on a 3-element array) return negative values instead of 0

4. **Internal consistency issue**: The function is used internally by pandas for array allocation and bounds checking (line 175 in the same file). Negative lengths could cause incorrect memory allocation or indexing errors downstream.

## Relevant Context

The bug occurs in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/indexers/utils.py` at line 316. The problematic calculation is:

```python
return (stop - start + step - 1) // step
```

This formula doesn't account for cases where the result should be clamped to non-negative values. The issue manifests in three scenarios:

1. When handling negative steps, the function swaps start and stop (lines 314-315) but the subsequent calculation can still produce negative results
2. When start > stop for positive steps, the formula yields negative values
3. When indices are out of bounds, the adjustments don't prevent negative results

The function is actively used within pandas, particularly in `check_setitem_lengths` (line 175), where incorrect length calculations could affect validation logic.

Reference: [pandas.core.indexers.utils source](https://github.com/pandas-dev/pandas/blob/main/pandas/core/indexers/utils.py#L290-L330)

## Proposed Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -313,7 +313,7 @@ def length_of_indexer(indexer, target=None) -> int:
         elif step < 0:
             start, stop = stop + 1, start + 1
             step = -step
-        return (stop - start + step - 1) // step
+        return max(0, (stop - start + step - 1) // step)
     elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
         if isinstance(indexer, list):
             indexer = np.array(indexer)
```