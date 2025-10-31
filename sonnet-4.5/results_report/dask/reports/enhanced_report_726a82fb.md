# Bug Report: dask.array.slicing.check_index Misleading Error Message for Boolean Arrays

**Target**: `dask.array.slicing.check_index`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `check_index` function incorrectly reports that a boolean array is "not long enough" when it's actually too long, providing misleading error messages that could confuse users during debugging.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from dask.array.slicing import check_index

@given(st.integers(min_value=1, max_value=100))
def test_check_index_error_message_accuracy(dim_size):
    too_long_array = np.array([True] * (dim_size + 1))

    try:
        check_index(0, too_long_array, dim_size)
        assert False, "Should have raised IndexError"
    except IndexError as e:
        error_msg = str(e)

        if "not long enough" in error_msg and too_long_array.size > dim_size:
            raise AssertionError(
                f"Error message says 'not long enough' but array size "
                f"{too_long_array.size} is greater than dimension {dim_size}"
            )

if __name__ == "__main__":
    test_check_index_error_message_accuracy()
```

<details>

<summary>
**Failing input**: `dim_size=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 10, in test_check_index_error_message_accuracy
    check_index(0, too_long_array, dim_size)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/array/slicing.py", line 920, in check_index
    raise IndexError(
    ...<2 lines>...
    )
IndexError: Boolean array with size 2 is not long enough for axis 0 with size 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 22, in <module>
    test_check_index_error_message_accuracy()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 6, in test_check_index_error_message_accuracy
    def test_check_index_error_message_accuracy(dim_size):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 16, in test_check_index_error_message_accuracy
    raise AssertionError(
    ...<2 lines>...
    )
AssertionError: Error message says 'not long enough' but array size 2 is greater than dimension 1
Falsifying example: test_check_index_error_message_accuracy(
    dim_size=1,
)
```
</details>

## Reproducing the Bug

```python
from dask.array.slicing import check_index
import numpy as np

# Test case: Boolean array too long for the dimension
bool_array = np.array([True, True, True])
dimension = 1

print(f"Boolean array size: {bool_array.size}")
print(f"Dimension size: {dimension}")
print(f"Array is {'too long' if bool_array.size > dimension else 'too short' if bool_array.size < dimension else 'correct size'}")
print()

try:
    check_index(0, bool_array, dimension)
    print("No error raised")
except IndexError as e:
    print(f"IndexError raised: {e}")
    print()
    print("Analysis:")
    if "not long enough" in str(e) and bool_array.size > dimension:
        print(f"ERROR: Message says 'not long enough' but array size {bool_array.size} > dimension size {dimension}")
        print("The array is actually TOO LONG, not too short!")
```

<details>

<summary>
IndexError: Boolean array is "not long enough" despite being too long
</summary>
```
Boolean array size: 3
Dimension size: 1
Array is too long

IndexError raised: Boolean array with size 3 is not long enough for axis 0 with size 1

Analysis:
ERROR: Message says 'not long enough' but array size 3 > dimension size 1
The array is actually TOO LONG, not too short!
```
</details>

## Why This Is A Bug

This violates the API's implicit contract to provide accurate and helpful error messages. When a boolean array has 3 elements and the dimension has size 1, mathematically the array is too long (3 > 1), not "not long enough". This misleading message could cause users to waste time trying to make their array longer when they actually need to make it shorter. While the function correctly identifies the size mismatch and raises an IndexError, the error message provides incorrect guidance about the nature of the problem.

## Relevant Context

The bug is present in the source code at `/home/npc/miniconda/lib/python3.13/site-packages/dask/array/slicing.py:919-923`. The function's own docstring examples (lines 904-907) actually demonstrate this bug, showing an error message that says "not long enough" when a size-3 array is used with a size-1 dimension.

The current implementation uses a single error message for both cases (array too short AND array too long):
- When array.size < dimension: Message says "not long enough" ✓ (correct)
- When array.size > dimension: Message says "not long enough" ✗ (incorrect - should say "too long")

This affects boolean indexing operations, which are commonly used in array operations for masking and filtering. The bug has been present in the codebase documentation examples, suggesting it may have existed for some time.

## Proposed Fix

```diff
--- a/dask/array/slicing.py
+++ b/dask/array/slicing.py
@@ -917,10 +917,15 @@ def check_index(axis, ind, dimension):
     elif is_arraylike(ind):
         if ind.dtype == bool:
             if ind.size != dimension:
-                raise IndexError(
-                    f"Boolean array with size {ind.size} is not long enough "
-                    f"for axis {axis} with size {dimension}"
-                )
+                if ind.size < dimension:
+                    msg = (
+                        f"Boolean array with size {ind.size} is not long enough "
+                        f"for axis {axis} with size {dimension}"
+                    )
+                else:
+                    msg = (
+                        f"Boolean array with size {ind.size} is too long "
+                        f"for axis {axis} with size {dimension}"
+                    )
+                raise IndexError(msg)
         elif (ind >= dimension).any() or (ind < -dimension).any():
             raise IndexError(
```