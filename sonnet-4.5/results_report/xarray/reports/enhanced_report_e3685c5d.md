# Bug Report: xarray.backends.chunks.build_grid_chunks Produces Invalid Chunk Sums

**Target**: `xarray.backends.chunks.build_grid_chunks`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `build_grid_chunks` function produces chunks that do not sum to the specified `size` parameter when `chunk_size > size`, violating the fundamental mathematical invariant that chunks must sum to the dimension size they represent.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from xarray.backends.chunks import build_grid_chunks

@settings(max_examples=500)
@given(
    size=st.integers(min_value=1, max_value=10000),
    chunk_size=st.integers(min_value=1, max_value=1000)
)
def test_build_grid_chunks_sum_invariant(size, chunk_size):
    chunks = build_grid_chunks(size=size, chunk_size=chunk_size, region=None)
    assert sum(chunks) == size

if __name__ == "__main__":
    test_build_grid_chunks_sum_invariant()
```

<details>

<summary>
**Failing input**: `size=1, chunk_size=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 14, in <module>
    test_build_grid_chunks_sum_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 5, in test_build_grid_chunks_sum_invariant
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 11, in test_build_grid_chunks_sum_invariant
    assert sum(chunks) == size
           ^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_build_grid_chunks_sum_invariant(
    size=1,
    chunk_size=2,
)
```
</details>

## Reproducing the Bug

```python
from xarray.backends.chunks import build_grid_chunks

# Test the failing case: size=1, chunk_size=2
result = build_grid_chunks(size=1, chunk_size=2, region=None)
print(f"build_grid_chunks(size=1, chunk_size=2, region=None)")
print(f"Result: {result}")
print(f"Sum of chunks: {sum(result)}")
print(f"Expected sum: 1")
print(f"Bug: The chunks sum to {sum(result)} instead of 1")
print()

# Additional test cases to show the pattern
test_cases = [
    (1, 1),   # Should work correctly
    (1, 10),  # Should fail
    (2, 3),   # Should fail
    (3, 10),  # Should fail
    (5, 5),   # Should work correctly
    (10, 3),  # Should work correctly
]

print("Additional test cases:")
for size, chunk_size in test_cases:
    chunks = build_grid_chunks(size=size, chunk_size=chunk_size, region=None)
    chunks_sum = sum(chunks)
    status = "✓" if chunks_sum == size else "✗"
    print(f"size={size:2}, chunk_size={chunk_size:2}: chunks={chunks}, sum={chunks_sum:2}, expected={size:2} {status}")
```

<details>

<summary>
AssertionError: Chunks sum to 3 instead of 1
</summary>
```
build_grid_chunks(size=1, chunk_size=2, region=None)
Result: (2, 1)
Sum of chunks: 3
Expected sum: 1
Bug: The chunks sum to 3 instead of 1

Additional test cases:
size= 1, chunk_size= 1: chunks=(1,), sum= 1, expected= 1 ✓
size= 1, chunk_size=10: chunks=(10, 1), sum=11, expected= 1 ✗
size= 2, chunk_size= 3: chunks=(3, 2), sum= 5, expected= 2 ✗
size= 3, chunk_size=10: chunks=(10, 3), sum=13, expected= 3 ✗
size= 5, chunk_size= 5: chunks=(5,), sum= 5, expected= 5 ✓
size=10, chunk_size= 3: chunks=(3, 3, 3, 1), sum=10, expected=10 ✓
```
</details>

## Why This Is A Bug

This function violates the fundamental mathematical invariant that chunks representing a dimension must sum to the dimension's size. The bug occurs specifically when `chunk_size > size`.

The function's logic on line 146 unconditionally creates a first chunk of size `chunk_size - (region_start % chunk_size)`, which equals `chunk_size` when `region_start=0`. This doesn't account for cases where `chunk_size` exceeds the total `size`.

When `size=1` and `chunk_size=2`:
1. First chunk is set to `2` (the full chunk_size)
2. Remaining size becomes `1 - 2 = -1`
3. Due to Python's modulo behavior with negatives, `-1 % 2 = 1`
4. This adds an erroneous "remainder" chunk of size `1`
5. Final result: `(2, 1)` with sum `3` instead of expected `1`

This violates the core contract that chunks should partition the data dimension exactly. In production usage via `grid_rechunk`, this could lead to:
- Out-of-bounds array access when chunks exceed array dimensions
- Data corruption in zarr storage backends
- Incorrect data slicing and misaligned chunk boundaries

## Relevant Context

The `build_grid_chunks` function is an internal utility in xarray's zarr backend implementation, located at `/xarray/backends/chunks.py:136-150`. It's called by `grid_rechunk` (line 163) to determine chunk sizes for zarr grid-based storage.

The function lacks documentation and has no explicit handling for the edge case where `chunk_size > size`. While this scenario is uncommon in typical usage (chunks are usually subdivisions of data), it can occur in practice when:
- Users specify large default chunk sizes that exceed small dimension sizes
- Automatic chunking algorithms produce suboptimal configurations
- Data is dynamically resized or subset after initial chunking

Related code: https://github.com/pydata/xarray/blob/main/xarray/backends/chunks.py#L136-L150

## Proposed Fix

The fix requires ensuring the first chunk never exceeds the total size by adding a `min()` constraint:

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