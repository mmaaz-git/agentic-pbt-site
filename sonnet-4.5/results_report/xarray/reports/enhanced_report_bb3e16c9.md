# Bug Report: xarray.backends.chunks.build_grid_chunks Returns Non-Zero Chunks for Empty Arrays

**Target**: `xarray.backends.chunks.build_grid_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `build_grid_chunks` function incorrectly returns chunks that sum to `chunk_size` instead of `0` when the input `size=0`, violating the fundamental partitioning invariant that the sum of chunks must equal the total size.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from xarray.backends.chunks import build_grid_chunks

@given(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=1, max_value=100)
)
def test_build_grid_chunks_sum_equals_size(size, chunk_size):
    chunks = build_grid_chunks(size=size, chunk_size=chunk_size, region=None)
    assert sum(chunks) == size, \
        f"sum(chunks)={sum(chunks)} != size={size} for chunks={chunks}"

# Run the test
test_build_grid_chunks_sum_equals_size()
```

<details>

<summary>
**Failing input**: `size=0, chunk_size=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 17, in <module>
    test_build_grid_chunks_sum_equals_size()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 8, in test_build_grid_chunks_sum_equals_size
    st.integers(min_value=0, max_value=1000),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 13, in test_build_grid_chunks_sum_equals_size
    assert sum(chunks) == size, \
           ^^^^^^^^^^^^^^^^^^^
AssertionError: sum(chunks)=1 != size=0 for chunks=(1,)
Falsifying example: test_build_grid_chunks_sum_equals_size(
    size=0,
    chunk_size=1,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/26/hypo.py:14
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.backends.chunks import build_grid_chunks

# Test case from the bug report: size=0, chunk_size=5
result = build_grid_chunks(size=0, chunk_size=5, region=None)
print(f"Test: build_grid_chunks(size=0, chunk_size=5, region=None)")
print(f"Result: {result}")
print(f"Sum of chunks: {sum(result)}")
print(f"Expected sum: 0")
print(f"Bug: sum(chunks)={sum(result)} != size=0")
```

<details>

<summary>
AssertionError: Chunks sum to 5 instead of 0
</summary>
```
Test: build_grid_chunks(size=0, chunk_size=5, region=None)
Result: (5,)
Sum of chunks: 5
Expected sum: 0
Bug: sum(chunks)=5 != size=0
```
</details>

## Why This Is A Bug

The `build_grid_chunks` function is designed to partition a given size into chunks of at most `chunk_size`. The fundamental mathematical invariant for any partitioning operation is that the sum of all partitions must equal the original value being partitioned. When `size=0`, the only valid partition is an empty tuple `()` or chunks that sum to 0.

The bug occurs at line 146 in `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/backends/chunks.py`:
```python
chunks_on_region = [chunk_size - (region_start % chunk_size)]
```

This line blindly creates an initial chunk without checking if `size` is smaller than the calculated chunk. For `size=0`, it creates a chunk of size `chunk_size`, which is incorrect.

The calling function `align_nd_chunks` (lines 20-25) explicitly validates this invariant:
```python
if sum(backend_chunks) != sum(var_chunks):
    raise ValueError(
        "The number of elements in the backend does not "
        "match the number of elements in the variable. "
        "This inconsistency should never occur at this stage."
    )
```

The error message states "This inconsistency should never occur at this stage," confirming that violating this invariant is a bug, not an expected edge case.

## Relevant Context

The `build_grid_chunks` function is called by `grid_rechunk` at line 164, which passes `sum(var_chunks)` as the size parameter. This sum could be 0 for empty arrays, which are valid in numpy and xarray.

When tracing through the code for `size=0, chunk_size=5, region=None`:
1. Line 142: `region = slice(0, 0, None)` (since size=0)
2. Line 144: `region_start = 0`
3. Line 146: `chunks_on_region = [5]` (calculated as `5 - (0 % 5) = 5`)
4. Line 147: Extension calculation: `(0 - 5) // 5 = -1`, so `[5] * (-1)` adds nothing
5. Line 148-149: Remainder check: `(0 - 5) % 5 = 0`, so no remainder chunk added
6. Line 150: Returns `(5,)` - **incorrect, should return `()`**

The function has no documentation, but its purpose is clear from usage and the mathematical semantics of chunking/partitioning operations.

## Proposed Fix

```diff
--- a/xarray/backends/chunks.py
+++ b/xarray/backends/chunks.py
@@ -143,7 +143,10 @@ def build_grid_chunks(

     region_start = region.start or 0
     # Generate the zarr chunks inside the region of this dim
-    chunks_on_region = [chunk_size - (region_start % chunk_size)]
+    if size == 0:
+        return ()
+
+    chunks_on_region = [min(size, chunk_size - (region_start % chunk_size))]
     chunks_on_region.extend([chunk_size] * ((size - chunks_on_region[0]) // chunk_size))
     if (size - chunks_on_region[0]) % chunk_size != 0:
         chunks_on_region.append((size - chunks_on_region[0]) % chunk_size)
```