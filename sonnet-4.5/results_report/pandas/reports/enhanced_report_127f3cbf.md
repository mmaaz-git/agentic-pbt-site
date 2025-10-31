# Bug Report: pandas.core.indexers.utils length_of_indexer Returns Negative Length for Empty Slices

**Target**: `pandas.core.indexers.utils.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer()` function returns negative values for empty slices (where start > stop with positive step), violating Python's fundamental contract that lengths are always non-negative.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.core.indexers.utils import length_of_indexer

@given(
    st.integers(min_value=-20, max_value=20),
    st.integers(min_value=-20, max_value=20),
    st.one_of(st.integers(min_value=-5, max_value=5).filter(lambda x: x != 0), st.none())
)
@settings(max_examples=500)
def test_length_of_indexer_slice_property(start, stop, step):
    target = list(range(50))
    slc = slice(start, stop, step)

    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length, f"Failed for slice({start}, {stop}, {step}): computed={computed_length}, actual={actual_length}"

if __name__ == "__main__":
    test_length_of_indexer_slice_property()
```

<details>

<summary>
**Failing input**: `slice(1, 0, None)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 20, in <module>
    test_length_of_indexer_slice_property()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 5, in test_length_of_indexer_slice_property
    st.integers(min_value=-20, max_value=20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 17, in test_length_of_indexer_slice_property
    assert computed_length == actual_length, f"Failed for slice({start}, {stop}, {step}): computed={computed_length}, actual={actual_length}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Failed for slice(1, 0, None): computed=-1, actual=0
Falsifying example: test_length_of_indexer_slice_property(
    start=1,
    stop=0,
    step=None,
)
```
</details>

## Reproducing the Bug

```python
from pandas.core.indexers.utils import length_of_indexer

# Demonstrate the bug with slice(1, 0, None)
target = list(range(50))
slc = slice(1, 0, None)

# Get the result from length_of_indexer
computed = length_of_indexer(slc, target)

# Get the actual result from Python's built-in slicing
actual = len(target[slc])

print(f"Demonstrating bug with empty slice where start > stop:")
print(f"Target: list(range(50))")
print(f"Slice: slice(1, 0, None)")
print()
print(f"length_of_indexer(slice(1, 0, None), target) = {computed}")
print(f"len(target[slice(1, 0, None)]) = {actual}")
print()
print(f"Expected: {actual}")
print(f"Got: {computed}")
print()

# Test additional cases to show pattern
test_cases = [
    slice(5, 2, None),
    slice(10, 3, None),
    slice(20, 0, None),
    slice(-3, -13, None),
]

print("Additional failing cases:")
for slc in test_cases:
    computed = length_of_indexer(slc, target)
    actual = len(target[slc])
    print(f"  slice{slc.start, slc.stop, slc.step}: computed={computed}, actual={actual}")

print()
print("Assertion check (will fail):")
slc = slice(1, 0, None)
computed = length_of_indexer(slc, target)
actual = len(target[slc])
assert computed == actual, f"Assertion failed: {computed} != {actual}"
```

<details>

<summary>
AssertionError: -1 != 0 for empty slice
</summary>
```
Demonstrating bug with empty slice where start > stop:
Target: list(range(50))
Slice: slice(1, 0, None)

length_of_indexer(slice(1, 0, None), target) = -1
len(target[slice(1, 0, None)]) = 0

Expected: 0
Got: -1

Additional failing cases:
  slice(5, 2, None): computed=-3, actual=0
  slice(10, 3, None): computed=-7, actual=0
  slice(20, 0, None): computed=-20, actual=0
  slice(-3, -13, None): computed=-10, actual=0

Assertion check (will fail):
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/repo.py", line 43, in <module>
    assert computed == actual, f"Assertion failed: {computed} != {actual}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Assertion failed: -1 != 0
```
</details>

## Why This Is A Bug

The `length_of_indexer()` function's docstring explicitly states it should "Return the expected length of target[indexer]". This creates a clear contract: the function must return the same length that would result from actually performing the indexing operation.

In Python, `len()` **never** returns negative values - this is a fundamental invariant of the language. When slicing a sequence with an empty slice (where start >= stop for positive step), Python returns an empty sequence with length 0, not a negative length.

The bug occurs at line 316 in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/indexers/utils.py`:

```python
return (stop - start + step - 1) // step
```

When start > stop (empty slice), this formula produces negative results:
- For `slice(1, 0, None)` with step=1: `(0 - 1 + 1 - 1) // 1 = -1 // 1 = -1`
- For `slice(5, 2, None)` with step=1: `(2 - 5 + 1 - 1) // 1 = -3 // 1 = -3`

This violates the function's contract and could cause downstream issues in pandas where:
1. Code assumes lengths are non-negative (common assumption)
2. Negative lengths are used in calculations, producing incorrect results
3. The function is used in `check_setitem_lengths()` at line 175, which could incorrectly validate slice assignments

## Relevant Context

The `length_of_indexer()` function is an internal pandas utility used throughout the codebase for indexing operations. While not part of the public API, it's critical for pandas' internal correctness.

The function handles various indexer types:
- Slices (where the bug occurs)
- Boolean arrays (correctly returns sum of True values)
- Integer arrays/lists (correctly returns their length)
- Ranges (has similar logic but may have the same issue)
- Scalar values (returns 1)

This bug only affects slice indexers when start > stop with positive step, or start < stop with negative step. These represent empty slices that should have length 0.

Relevant pandas source code location:
- File: `/pandas/core/indexers/utils.py`
- Function: `length_of_indexer` (lines 290-329)
- Bug location: Line 316

## Proposed Fix

The fix is straightforward - ensure the result is never negative by wrapping it in `max(0, ...)`:

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

This ensures the function always returns non-negative values, matching Python's slice behavior while maintaining backward compatibility for all valid (non-empty) slices.