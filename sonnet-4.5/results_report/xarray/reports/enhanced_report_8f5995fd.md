# Bug Report: xarray.backends.chunks.build_grid_chunks Violates Chunking Invariant

**Target**: `xarray.backends.chunks.build_grid_chunks`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `build_grid_chunks` function returns chunks that sum to more than the specified `size` when `chunk_size > size`, violating the fundamental invariant that chunks must partition (not exceed) the data size.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from xarray.backends.chunks import build_grid_chunks

@given(
    size=st.integers(min_value=1, max_value=1000),
    chunk_size=st.integers(min_value=1, max_value=100),
)
@example(size=1, chunk_size=2)  # Known minimal failing case
def test_build_grid_chunks_sum_equals_size(size, chunk_size):
    """Test that build_grid_chunks returns chunks that sum to exactly the input size."""
    chunks = build_grid_chunks(size, chunk_size, region=None)

    # The fundamental invariant: chunks must partition the data, meaning their sum equals size
    assert sum(chunks) == size, f"Chunks {chunks} sum to {sum(chunks)}, expected {size}"

    # Additional validation: all chunks should be positive
    assert all(c > 0 for c in chunks), f"All chunks should be positive, got {chunks}"

    # Additional validation: no chunk except possibly the last should exceed chunk_size
    for i, c in enumerate(chunks[:-1]):
        assert c <= chunk_size, f"Chunk {i} has size {c} which exceeds chunk_size {chunk_size}"

if __name__ == "__main__":
    # Run the property test
    test_build_grid_chunks_sum_equals_size()
```

<details>

<summary>
**Failing input**: `size=1, chunk_size=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 25, in <module>
    test_build_grid_chunks_sum_equals_size()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 5, in test_build_grid_chunks_sum_equals_size
    size=st.integers(min_value=1, max_value=1000),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 14, in test_build_grid_chunks_sum_equals_size
    assert sum(chunks) == size, f"Chunks {chunks} sum to {sum(chunks)}, expected {size}"
           ^^^^^^^^^^^^^^^^^^^
AssertionError: Chunks (2, 1) sum to 3, expected 1
Falsifying explicit example: test_build_grid_chunks_sum_equals_size(
    size=1,
    chunk_size=2,
)
```
</details>

## Reproducing the Bug

```python
from xarray.backends.chunks import build_grid_chunks

# Test case 1: size=1, chunk_size=2
result = build_grid_chunks(size=1, chunk_size=2, region=None)
print(f"Test case 1: size=1, chunk_size=2")
print(f"Result: {result}")
print(f"Sum: {sum(result)}")
print(f"Expected sum: 1")
print(f"BUG: Sum exceeds size by {sum(result) - 1}")
print()

# Test case 2: size=5, chunk_size=10
result = build_grid_chunks(size=5, chunk_size=10, region=None)
print(f"Test case 2: size=5, chunk_size=10")
print(f"Result: {result}")
print(f"Sum: {sum(result)}")
print(f"Expected sum: 5")
print(f"BUG: Sum exceeds size by {sum(result) - 5}")
print()

# Test case 3: size=3, chunk_size=10
result = build_grid_chunks(size=3, chunk_size=10, region=None)
print(f"Test case 3: size=3, chunk_size=10")
print(f"Result: {result}")
print(f"Sum: {sum(result)}")
print(f"Expected sum: 3")
print(f"BUG: Sum exceeds size by {sum(result) - 3}")
print()

# Test case 4 (working case): size=10, chunk_size=3
result = build_grid_chunks(size=10, chunk_size=3, region=None)
print(f"Test case 4 (should work): size=10, chunk_size=3")
print(f"Result: {result}")
print(f"Sum: {sum(result)}")
print(f"Expected sum: 10")
print(f"CORRECT: Sum equals size" if sum(result) == 10 else f"BUG: Sum differs by {sum(result) - 10}")
```

<details>

<summary>
Bug demonstrated: chunks sum exceeds data size
</summary>
```
Test case 1: size=1, chunk_size=2
Result: (2, 1)
Sum: 3
Expected sum: 1
BUG: Sum exceeds size by 2

Test case 2: size=5, chunk_size=10
Result: (10, 5)
Sum: 15
Expected sum: 5
BUG: Sum exceeds size by 10

Test case 3: size=3, chunk_size=10
Result: (10, 3)
Sum: 13
Expected sum: 3
BUG: Sum exceeds size by 10

Test case 4 (should work): size=10, chunk_size=3
Result: (3, 3, 3, 1)
Sum: 10
Expected sum: 10
CORRECT: Sum equals size
```
</details>

## Why This Is A Bug

The `build_grid_chunks` function is designed to partition a region of `size` elements into chunks of approximately `chunk_size`. The fundamental invariant of any chunking operation is that **chunks must partition the data** - meaning the sum of all chunks must equal exactly the total size of the data being chunked.

When `chunk_size > size`, the function violates this invariant by returning chunks that sum to more than the input size. Specifically:

1. **Line 146** unconditionally sets the first chunk to `chunk_size - (region_start % chunk_size)`, which equals `chunk_size` when `region_start=0`
2. This first chunk can exceed the total `size` when `chunk_size > size`
3. The algorithm then incorrectly adds additional chunks based on negative remainder calculations in Python

This violates the chunking contract and contradicts the function's purpose as used in `grid_rechunk` (line 164 of chunks.py), where it's called with `sum(var_chunks)` expecting it to properly partition that exact size.

## Relevant Context

- **Usage Context**: This function is used in production by `grid_rechunk` which is called from `xarray.backends.zarr` when writing data with aligned chunks
- **Test Coverage Gap**: The existing test suite in `test_backends_chunks.py` only tests cases where `size >= chunk_size`, completely missing this edge case
- **Potential Impact**: If triggered in production, this could cause:
  - Array out-of-bounds errors when accessing data based on these chunks
  - Data corruption when writing to Zarr stores
  - Incorrect computations in downstream operations that rely on correct chunking
- **Python-specific behavior**: The bug is exacerbated by Python's modulo operation on negative numbers, where `-1 % 2 = 1`, causing the algorithm to add an extra chunk

Links:
- Source code: `/home/npc/miniconda/lib/python3.13/site-packages/xarray/backends/chunks.py:136-150`
- Tests: `/home/npc/miniconda/lib/python3.13/site-packages/xarray/tests/test_backends_chunks.py:9-25`
- Usage in zarr backend: `/home/npc/miniconda/lib/python3.13/site-packages/xarray/backends/zarr.py`

## Proposed Fix

```diff
--- a/xarray/backends/chunks.py
+++ b/xarray/backends/chunks.py
@@ -143,7 +143,7 @@ def build_grid_chunks(

     region_start = region.start or 0
     # Generate the zarr chunks inside the region of this dim
-    chunks_on_region = [chunk_size - (region_start % chunk_size)]
+    chunks_on_region = [min(size, chunk_size - (region_start % chunk_size))]
     chunks_on_region.extend([chunk_size] * ((size - chunks_on_region[0]) // chunk_size))
     if (size - chunks_on_region[0]) % chunk_size != 0:
         chunks_on_region.append((size - chunks_on_region[0]) % chunk_size)
```

This fix ensures the first chunk never exceeds the total `size` by using `min(size, ...)`, which correctly handles the edge case while preserving all existing behavior for normal cases where `size >= chunk_size`.