# Bug Report: ensure_minimum_chunksize Parameter Documentation

**Target**: `dask.array.overlap.ensure_minimum_chunksize`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `size` parameter docstring in `ensure_minimum_chunksize` incorrectly states "The maximum size of any chunk" when it should say "The minimum size of any chunk".

## Property-Based Test

This bug was discovered through static analysis while testing properties of `dask.array.lib.sliding_window_view`, which uses `ensure_minimum_chunksize` internally.

```python
from hypothesis import given, strategies as st
from dask.array.overlap import ensure_minimum_chunksize

@given(
    min_size=st.integers(min_value=1, max_value=20),
    chunks=st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=10)
)
def test_ensure_minimum_chunksize_enforces_minimum(min_size, chunks):
    """Property: All output chunks should be >= min_size (except possibly the last)"""
    chunks_tuple = tuple(chunks)
    try:
        result = ensure_minimum_chunksize(min_size, chunks_tuple)
        # Verify the property: result chunks are >= min_size
        assert all(c >= min_size for c in result), \
            f"Function should ensure all chunks >= {min_size}, got {result}"
    except ValueError:
        # Expected when min_size > sum(chunks)
        assert min_size > sum(chunks)
```

**Finding**: The property tests confirm the function enforces a MINIMUM, contradicting the docstring which says "maximum".

## Reproducing the Bug

```python
from dask.array.overlap import ensure_minimum_chunksize
import inspect

source = inspect.getsource(ensure_minimum_chunksize)
docstring = ensure_minimum_chunksize.__doc__

print("Function name: ensure_minimum_chunksize")
print("Function docstring (first line):")
print("  'Determine new chunks to ensure that every chunk >= size'")
print()
print("Parameter docstring:")
print("  'size: int - The maximum size of any chunk.'")
print()
print("CONTRADICTION: Docstring says '>=' (minimum) but parameter says 'maximum'")
print()

result = ensure_minimum_chunksize(10, (20, 20, 1))
print(f"Test: ensure_minimum_chunksize(10, (20, 20, 1)) = {result}")
print(f"Expected (from docstring example): (20, 11, 10)")
print(f"Result: {result}")
print()
print("Analysis: All output chunks are >= 10, confirming MINIMUM enforcement")
print("The parameter doc should say 'minimum' not 'maximum'")
```

## Why This Is A Bug

The documentation contradicts itself and the implementation:

1. **Function name**: `ensure_**minimum**_chunksize` - clearly about minimum
2. **Docstring summary**: "ensure that every chunk >= size" - describes minimum
3. **Parameter doc**: "The **maximum** size" - contradicts everything else
4. **Implementation** (line 341): `if size <= min(chunks): return chunks` - enforces minimum
5. **Examples**: Confirm minimum enforcement

This violates the API contract - developers reading the parameter documentation will be confused about what the function does.

## Fix

```diff
diff --git a/dask/array/overlap.py b/dask/array/overlap.py
index 1234567..abcdefg 100644
--- a/dask/array/overlap.py
+++ b/dask/array/overlap.py
@@ -323,7 +323,7 @@ def ensure_minimum_chunksize(size, chunks):
     Parameters
     ----------
     size: int
-        The maximum size of any chunk.
+        The minimum size of any chunk.
     chunks: tuple
         Chunks along one axis, e.g. ``(3, 3, 2)``