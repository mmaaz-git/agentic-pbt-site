# Bug Report: xarray.backends.chunks build_grid_chunks Violates Sum Invariant

**Target**: `xarray.backends.chunks.build_grid_chunks`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `build_grid_chunks` function returns chunks that sum to more than the input `size` parameter when `size < chunk_size`, violating the fundamental mathematical invariant that chunks must sum to exactly the size they're dividing.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.backends.chunks import build_grid_chunks

@given(
    size=st.integers(min_value=1, max_value=1000),
    chunk_size=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=100)
def test_build_grid_chunks_sum(size, chunk_size):
    chunks = build_grid_chunks(size, chunk_size)

    assert sum(chunks) == size, \
        f"Sum of chunks {sum(chunks)} != size {size} (chunks={chunks}, chunk_size={chunk_size})"

    assert all(c > 0 for c in chunks), \
        f"All chunks should be positive, got {chunks}"

if __name__ == "__main__":
    test_build_grid_chunks_sum()
```

<details>

<summary>
**Failing input**: `size=1, chunk_size=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 19, in <module>
    test_build_grid_chunks_sum()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 5, in test_build_grid_chunks_sum
    size=st.integers(min_value=1, max_value=1000),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 12, in test_build_grid_chunks_sum
    assert sum(chunks) == size, \
           ^^^^^^^^^^^^^^^^^^^
AssertionError: Sum of chunks 3 != size 1 (chunks=(2, 1), chunk_size=2)
Falsifying example: test_build_grid_chunks_sum(
    size=1,
    chunk_size=2,
)
```
</details>

## Reproducing the Bug

```python
from xarray.backends.chunks import build_grid_chunks

# Test case that demonstrates the bug
size = 1
chunk_size = 2

print(f"Testing build_grid_chunks with size={size}, chunk_size={chunk_size}")
result = build_grid_chunks(size=size, chunk_size=chunk_size)
print(f"Result: {result}")
print(f"Sum of chunks: {sum(result)}")
print(f"Expected sum: {size}")
print(f"ERROR: Sum of chunks ({sum(result)}) != size ({size})")

# Verify the assertion fails
try:
    assert sum(result) == size
    print("Assertion passed (unexpected)")
except AssertionError:
    print("AssertionError: sum(result) != size")
```

<details>

<summary>
Output showing chunks sum to 3 instead of 1
</summary>
```
Testing build_grid_chunks with size=1, chunk_size=2
Result: (2, 1)
Sum of chunks: 3
Expected sum: 1
ERROR: Sum of chunks (3) != size (1)
AssertionError: sum(result) != size
```
</details>

## Why This Is A Bug

This violates the fundamental mathematical contract that when dividing a size into chunks, the chunks must sum to exactly that size. The bug occurs systematically whenever `size < chunk_size`:

1. **Contract Enforcement**: The `align_nd_chunks` function at line 20 of `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/backends/chunks.py` explicitly validates this invariant:
   ```python
   if sum(backend_chunks) != sum(var_chunks):
       raise ValueError(
           "The number of elements in the backend does not "
           "match the number of elements in the variable. "
           "This inconsistency should never occur at this stage."
       )
   ```

2. **Root Cause**: At line 146, the first chunk is computed as `chunk_size - (region_start % chunk_size)`. When `region_start=0` and `size=1, chunk_size=2`:
   - First chunk = `2 - (0 % 2)` = `2` (but size is only 1!)
   - Remaining = `(1 - 2) % 2` = `1` (Python's modulo with negative numbers)
   - Result: `(2, 1)` with sum `3` instead of expected `1`

3. **Usage Context**: The function is called from `grid_rechunk` at line 164 with `sum(var_chunks)` as the size parameter, establishing that returned chunks MUST sum to this exact value.

## Relevant Context

The bug affects all cases where a dimension size is smaller than the desired chunk size, which is common in scientific data:
- Single timestep dimensions (size=1) with larger chunk preferences
- Small vertical levels (e.g., size=5) with larger optimal chunk sizes (e.g., 10)
- This could lead to data corruption when writing to Zarr format with `align_chunks=True`

Testing reveals the bug is systematic for `size < chunk_size`:
- size=1, chunk_size=2 → chunks=(2,1), sum=3 (expected 1)
- size=2, chunk_size=3 → chunks=(3,2), sum=5 (expected 2)
- size=10, chunk_size=20 → chunks=(20,10), sum=30 (expected 10)
- size=99, chunk_size=100 → chunks=(100,99), sum=199 (expected 99)

Function location: `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/backends/chunks.py:136-150`

## Proposed Fix

```diff
def build_grid_chunks(
    size: int,
    chunk_size: int,
    region: slice | None = None,
) -> tuple[int, ...]:
    if region is None:
        region = slice(0, size)

    region_start = region.start or 0
    # Generate the zarr chunks inside the region of this dim
-    chunks_on_region = [chunk_size - (region_start % chunk_size)]
+    first_chunk = chunk_size - (region_start % chunk_size)
+    chunks_on_region = [min(first_chunk, size)]
    chunks_on_region.extend([chunk_size] * ((size - chunks_on_region[0]) // chunk_size))
    if (size - chunks_on_region[0]) % chunk_size != 0:
        chunks_on_region.append((size - chunks_on_region[0]) % chunk_size)
    return tuple(chunks_on_region)
```