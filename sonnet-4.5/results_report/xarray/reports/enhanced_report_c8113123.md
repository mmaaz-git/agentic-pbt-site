# Bug Report: xarray.backends.chunks.build_grid_chunks Returns Chunks That Exceed Input Size

**Target**: `xarray.backends.chunks.build_grid_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `build_grid_chunks` function incorrectly calculates chunk sizes when `chunk_size > size`, returning chunks that sum to more than the specified `size` parameter, violating its fundamental mathematical invariant.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.backends.chunks import build_grid_chunks

@given(
    size=st.integers(min_value=1, max_value=1000),
    chunk_size=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=100)
def test_build_grid_chunks_preserves_size(size, chunk_size):
    result = build_grid_chunks(size, chunk_size)
    assert sum(result) == size, f"Chunks {result} sum to {sum(result)}, expected {size}"

if __name__ == "__main__":
    # Run the test and let Hypothesis find and report the failing example
    test_build_grid_chunks_preserves_size()
```

<details>

<summary>
**Failing input**: `size=1, chunk_size=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 15, in <module>
    test_build_grid_chunks_preserves_size()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 5, in test_build_grid_chunks_preserves_size
    size=st.integers(min_value=1, max_value=1000),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 11, in test_build_grid_chunks_preserves_size
    assert sum(result) == size, f"Chunks {result} sum to {sum(result)}, expected {size}"
           ^^^^^^^^^^^^^^^^^^^
AssertionError: Chunks (2, 1) sum to 3, expected 1
Falsifying example: test_build_grid_chunks_preserves_size(
    size=1,
    chunk_size=2,
)
```
</details>

## Reproducing the Bug

```python
from xarray.backends.chunks import build_grid_chunks

# Test case that demonstrates the bug
result = build_grid_chunks(size=1, chunk_size=2)
print(f"Result: {result}")
print(f"Sum of chunks: {sum(result)}")
print(f"Expected sum: 1")

# This should be True but isn't
print(f"Sum equals size: {sum(result) == 1}")

# This assertion will fail
assert sum(result) == 1, f"Chunks sum to {sum(result)} instead of 1"
```

<details>

<summary>
AssertionError: Chunks sum to 3 instead of 1
</summary>
```
Result: (2, 1)
Sum of chunks: 3
Expected sum: 1
Sum equals size: False
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/repo.py", line 13, in <module>
    assert sum(result) == 1, f"Chunks sum to {sum(result)} instead of 1"
           ^^^^^^^^^^^^^^^^
AssertionError: Chunks sum to 3 instead of 1
```
</details>

## Why This Is A Bug

This violates the fundamental contract of `build_grid_chunks` that the returned chunks must sum to exactly `size`. This invariant is critical for the correctness of xarray's chunking system and is enforced elsewhere in the codebase.

Specifically, the `align_nd_chunks` function (lines 20-25 in `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/backends/chunks.py`) validates this invariant:

```python
if sum(backend_chunks) != sum(var_chunks):
    raise ValueError(
        "The number of elements in the backend does not "
        "match the number of elements in the variable. "
        "This inconsistency should never occur at this stage."
    )
```

The bug occurs in line 146 of `build_grid_chunks` when calculating the first chunk size:
```python
chunks_on_region = [chunk_size - (region_start % chunk_size)]
```

When `region_start % chunk_size == 0` (which is true when `region_start=0` or when aligned to chunk boundaries), this becomes `chunk_size`. If `chunk_size > size`, the first chunk becomes larger than the total size we're trying to partition, leading to incorrect results.

## Relevant Context

The `build_grid_chunks` function is used by `grid_rechunk` (lines 163-167) to generate backend chunks for zarr arrays. When working with small arrays or regions that are smaller than the specified chunk size, this bug causes the function to return chunks that exceed the array boundaries.

This is particularly problematic in distributed computing scenarios where Dask relies on correct chunk alignment to avoid data corruption during parallel writes. The validation functions like `validate_grid_chunks_alignment` expect chunks to respect array boundaries and would fail if this bug manifests in production.

## Proposed Fix

```diff
--- a/xarray/backends/chunks.py
+++ b/xarray/backends/chunks.py
@@ -143,7 +143,8 @@ def build_grid_chunks(

     region_start = region.start or 0
     # Generate the zarr chunks inside the region of this dim
-    chunks_on_region = [chunk_size - (region_start % chunk_size)]
+    first_chunk = min(size, chunk_size - (region_start % chunk_size))
+    chunks_on_region = [first_chunk]
     chunks_on_region.extend([chunk_size] * ((size - chunks_on_region[0]) // chunk_size))
     if (size - chunks_on_region[0]) % chunk_size != 0:
         chunks_on_region.append((size - chunks_on_region[0]) % chunk_size)
```